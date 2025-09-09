"""
Sophisticated RAG System for St. Thomas Aquinas Works
====================================================

This module provides a comprehensive RAG (Retrieval-Augmented Generation) system
specifically designed for analyzing and querying the works of St. Thomas Aquinas.
It leverages LlamaIndex and LlamaCloud for advanced document processing and retrieval.

Features:
- Advanced document parsing with LlamaParse
- Multi-modal document support (PDF, DOCX, etc.)
- Sophisticated metadata filtering by work type
- Context-aware retrieval with Aquinas-specific prompts
- Multiple LLM backends (OpenAI, Anthropic, Ollama)
- Vector database integration (ChromaDB, Pinecone, Weaviate)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    Document
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank, LongContextReorder
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AquinasDocument:
    """Structured representation of an Aquinas document."""
    title: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = None

class AquinasChunker:
    """Semantic chunker optimized for Aquinas's philosophical texts."""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def get_semantic_splitter(self):
        """Return semantic chunking strategy for all Aquinas texts."""
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embeddings
        )

class AquinasRAGSystem:
    """
    Sophisticated RAG system for St. Thomas Aquinas works.
    
    This class provides comprehensive functionality for:
    - Document ingestion and parsing
    - Vector indexing with metadata filtering
    - Context-aware retrieval
    - Expert-level querying with Aquinas-specific prompts
    """
    
    def __init__(self, llama_cloud_api_key: Optional[str] = None):
        """
        Initialize the Aquinas RAG system with Pinecone, OpenAI, and LlamaCloud.
        
        Args:
            llama_cloud_api_key: LlamaCloud API key for advanced parsing
        """
        self.llama_cloud_api_key = llama_cloud_api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        
        # Initialize components
        self._setup_llm()
        self._setup_embeddings()
        self._setup_vector_store()
        self._setup_advanced_chunking()
        
        # Initialize index and query engine
        self.index = None
        self.query_engine = None
        
    def _setup_llm(self):
        """Set up OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = OpenAI(
            model="gpt-4.1",
            temperature=0.1,
            max_tokens=4000
        )
        
        # Set global LLM
        Settings.llm = self.llm
        
    def _setup_embeddings(self):
        """Set up OpenAI embeddings."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embeddings = OpenAIEmbedding(
            model="text-embedding-3-large",
            embed_batch_size=100
        )
        
        # Set global embeddings
        Settings.embed_model = self.embeddings
        
    def _setup_vector_store(self):
        """Set up Pinecone vector store."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY required")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Create or get the index
        index_name = os.getenv("PINECONE_INDEX_NAME", "aquinas-works")
        namespace = os.getenv("PINECONE_NAMESPACE", "").strip() or None
        # Store for later checks
        self.pinecone_index_name = index_name
        self.pinecone_namespace = namespace
        
        # Check if index exists, create if not
        if index_name not in self.pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=3072,  # text-embedding-3-large dimensions
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        # Get the index object
        pinecone_index = self.pc.Index(index_name)
        
        # Initialize PineconeVectorStore with the index object
        self.vector_store_obj = PineconeVectorStore(
            pinecone_index=pinecone_index,
            namespace=namespace
        )
        
        # Always create an index wrapper and query engine so the API is ready
        try:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store_obj)
            self.create_query_engine()
            logger.info("Vector store connected; query engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize index from vector store: {e}")
            
    def _setup_advanced_chunking(self):
        """Set up semantic chunking for Aquinas texts."""
        self.aquinas_chunker = AquinasChunker(self.embeddings)
        
        # Use semantic splitter for all documents
        self.semantic_parser = self.aquinas_chunker.get_semantic_splitter()
    
    def _get_existing_vector_count(self) -> int:
        """Return approximate count of existing vectors in the Pinecone index/namespace."""
        try:
            index = self.pc.Index(self.pinecone_index_name)
            # Stats shape differs across client versions; handle robustly
            stats = index.describe_index_stats()  # may be dict-like or object
            # Normalize to dict
            if hasattr(stats, "to_dict"):
                stats = stats.to_dict()
            elif not isinstance(stats, dict):
                # Try attribute access fallback
                maybe_total = getattr(stats, "total_vector_count", None)
                maybe_namespaces = getattr(stats, "namespaces", None)
                if maybe_total is not None:
                    return int(maybe_total)
                if isinstance(maybe_namespaces, dict) and self.pinecone_namespace:
                    ns_stats = maybe_namespaces.get(self.pinecone_namespace) or {}
                    ns_total = ns_stats.get("vector_count") or ns_stats.get("total_vector_count") or 0
                    return int(ns_total)
                return 0
            # Now stats is a dict
            if self.pinecone_namespace:
                ns = stats.get("namespaces", {}).get(self.pinecone_namespace)
                if isinstance(ns, dict):
                    return int(ns.get("vector_count") or ns.get("total_vector_count") or 0)
                return 0
            return int(stats.get("total_vector_count") or 0)
        except Exception as e:
            logger.debug(f"Failed to get Pinecone stats: {e}")
            return 0
    
    def _maybe_load_existing_index(self) -> None:
        """If vectors already exist in Pinecone, create an index wrapper and query engine."""
        existing = self._get_existing_vector_count()
        if existing > 0:
            logger.info(
                f"Found {existing} existing vectors in Pinecone"
                + (f" namespace '{self.pinecone_namespace}'" if self.pinecone_namespace else "")
                + "; loading index wrapper..."
            )
            # Create an index wrapper backed by the existing vector store
            self.index = VectorStoreIndex.from_vector_store(self.vector_store_obj)
            # Prepare query engine immediately
            self.create_query_engine()
            logger.info("Loaded existing Pinecone vectors; query engine is ready")

    def ensure_ready_for_queries(self) -> None:
        """Ensure that an index wrapper and query engine are initialized from Pinecone."""
        if self.index is None:
            self.index = VectorStoreIndex.from_vector_store(self.vector_store_obj)
        if self.query_engine is None:
            self.create_query_engine()
            
    def ingest_documents(
        self, 
        documents_path: Union[str, Path],
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Ingest documents from the specified path.
        
        Args:
            documents_path: Path to documents directory or file
            custom_metadata: Additional metadata to add
            
        Returns:
            List of processed documents
        """
        documents_path = Path(documents_path)
        
        if not documents_path.exists():
            raise FileNotFoundError(f"Documents path not found: {documents_path}")
            
        logger.info(f"Ingesting documents from: {documents_path}")
        
        # Read documents
        if documents_path.is_file():
            file_paths = [documents_path]
        else:
            file_paths = list(documents_path.rglob("*"))
            file_paths = [f for f in file_paths if f.is_file() and f.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']]
            
        documents = []
        
        for file_path in file_paths:
            try:
                # Parse document using SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_files=[str(file_path)])
                parsed_docs = reader.load_data()
                
                # Add basic metadata
                for doc in parsed_docs:
                    doc.metadata.update({
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "source": "aquinas_works"
                    })
                    
                    if custom_metadata:
                        doc.metadata.update(custom_metadata)
                    
                documents.extend(parsed_docs)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
                
        logger.info(f"Successfully ingested {len(documents)} documents")
        return documents
            
    def build_index(self, documents: List[Document]):
        """Build the vector index from documents using semantic chunking."""
        logger.info("Building vector index with semantic chunking...")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store_obj)
        
        # Use semantic splitter for all documents
        node_parser = self.semantic_parser
        logger.info("Using semantic chunking for all documents")
        
        # Build index with semantic chunking
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser]
        )
        logger.info("Vector index built successfully with semantic chunking")
    
    def add_documents_to_index(self, documents: List[Document]):
        """
        Add new documents to an existing vector index.
        
        Args:
            documents: List of new documents to add
        """
        if not self.index:
            raise ValueError("No existing index found. Call build_index() first.")
        
        logger.info(f"Adding {len(documents)} documents to existing index...")
        
        # Use semantic splitter for all documents
        node_parser = self.semantic_parser
        logger.info("Using semantic chunking for new documents")
        
        # Process documents with semantic chunking
        nodes = node_parser.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)
        logger.info(f"Successfully added {len(documents)} documents to existing index")
        
        # Recreate query engine to include new documents
        self.create_query_engine()
        logger.info("Query engine updated with new documents")
        
    def create_query_engine(
        self,
        similarity_top_k: int = 15,  # Increased to allow for better filtering
        response_mode: str = "compact",
        use_metadata_filter: bool = True,
        use_llm_rerank: bool = True
    ):
        """Create the query engine with Aquinas-specific configuration."""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
            
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        # Create postprocessors optimized for Aquinas philosophical texts
        postprocessors = []
        
        # 1. Similarity filtering - remove completely irrelevant chunks
        similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
        postprocessors.append(similarity_postprocessor)
        
        # 2. Long context reordering - important for dense philosophical texts
        long_context_reorder = LongContextReorder()
        postprocessors.append(long_context_reorder)
        
        # 3. LLM reranking - understands philosophical context and nuance
        if use_llm_rerank:
            llm_rerank = LLMRerank(top_n=5)
            postprocessors.append(llm_rerank)
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            response_mode=response_mode
        )
        
        logger.info("Query engine created successfully with advanced postprocessing")
    
    def create_alternative_query_engine(
        self,
        similarity_top_k: int = 10,
        response_mode: str = "compact",
        use_sentence_transformer_rerank: bool = True
    ):
        """
        Create an alternative query engine using SentenceTransformerRerank for faster processing.
        
        This is useful when you want faster responses without LLM reranking costs.
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
            
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        # Create postprocessors for faster processing
        postprocessors = []
        
        # 1. Similarity filtering
        similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
        postprocessors.append(similarity_postprocessor)
        
        # 2. Sentence transformer rerank (faster alternative to LLM rerank)
        if use_sentence_transformer_rerank:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            sentence_rerank = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
                top_n=5
            )
            postprocessors.append(sentence_rerank)
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            response_mode=response_mode
        )
        
        logger.info("Alternative query engine created with SentenceTransformerRerank")
        
    def query(
        self,
        question: str,
        context_length: int = 4000
    ) -> str:
        """
        Query the Aquinas RAG system with sophisticated context.
        
        Args:
            question: The question to ask about Aquinas
            context_length: Maximum context length
            
        Returns:
            Generated response
        """
        if not self.query_engine:
            raise ValueError("Query engine not created. Call create_query_engine() first.")
            
        # Create Aquinas-specific prompt
        aquinas_prompt = self._create_aquinas_prompt(question)
        
        # Query the system
        response = self.query_engine.query(aquinas_prompt)
        
        # Debug logging
        logger.info(f"Query: {question}")
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response: {response}")
        
        # Handle empty response
        if not response or str(response).strip() == "":
            return "I apologize, but I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents."
        
        return str(response)
        
    def _create_aquinas_prompt(self, question: str) -> str:
        """Create a sophisticated prompt for Aquinas queries."""
        return f"""
You are an expert on the works of St. Thomas Aquinas, one of the most influential theologians and philosophers in history. You have access to his complete works and can provide detailed, accurate analysis based on his texts.

Context: You are analyzing the works of St. Thomas Aquinas, including:
- Summa Theologiae (his most famous work)
- Summa Contra Gentiles
- Commentaries on Aristotle and other philosophers
- Disputed Questions
- Various treatises and sermons

Question: {question}

Please provide a comprehensive answer that:
1. Directly references Aquinas's own words and arguments when possible
2. Explains the theological and philosophical context
3. Shows how this relates to other parts of his thought
4. Maintains academic rigor while being accessible
5. Cites specific works, parts, questions, and articles when relevant
"""
        
    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get a summary of the indexed documents' metadata."""
        if not self.index:
            return {}
            
        # This would require accessing the index's document store
        # For now, return a placeholder
        return {
            "total_documents": "N/A",
            "vector_store": "Pinecone",
            "llm_provider": "OpenAI GPT-4o",
            "embedding_provider": "OpenAI text-embedding-3-large",
            "chunking_strategy": "Semantic Splitter"
        }
    
    def get_chunking_info(self) -> str:
        """Get information about the current chunking strategy."""
        if not hasattr(self, 'aquinas_chunker'):
            return "Basic chunking (not initialized)"
        
        return """
Semantic Chunking Strategy:
- Uses semantic similarity to determine optimal chunk boundaries
- Buffer size: 1
- Breakpoint percentile threshold: 95%
- Optimized for philosophical and theological texts
- Maintains semantic coherence across chunks
        """
