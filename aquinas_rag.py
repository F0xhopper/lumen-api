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
import time
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
        
        # Rate limiting configuration
        self.embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "10"))
        self.embed_delay = float(os.getenv("EMBED_DELAY", "0.5"))  # seconds between batches
        
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
            embed_batch_size=self.embed_batch_size,
            timeout=60.0,  # Increased timeout for slower requests
            max_retries=3  # Add retry logic
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
        index_name = os.getenv("PINECONE_INDEX_NAME", "aquinas-works-testing-page-number")
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
        
    def _rate_limit_delay(self):
        """Add a small delay to avoid rate limiting."""
        if self.embed_delay > 0:
            time.sleep(self.embed_delay)
    
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
                # Add rate limiting delay between file processing
                self._rate_limit_delay()
                
                # Parse document using SimpleDirectoryReader
                reader = SimpleDirectoryReader(input_files=[str(file_path)])
                parsed_docs = reader.load_data()
                
                # Add basic metadata
                for i,doc in enumerate(parsed_docs):
                    doc.metadata.update({
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "leaf_number": i + 1,
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
            show_progress=True,
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
        
        # Create query engine with basic response
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
        
        # Create query engine with basic response
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=postprocessors,
            response_mode=response_mode
        )
        
        logger.info("Alternative query engine created with SentenceTransformerRerank")
        
    def query(
        self,
        question: str,
        context_length: int = 4000,
        retrieve_passages: bool = True
    ) -> str:
        """
        Query the Aquinas RAG system with sophisticated context.
        
        Args:
            question: The question to ask about Aquinas
            context_length: Maximum context length
            retrieve_passages: Whether to retrieve and include passages in the prompt
            
        Returns:
            Generated response
        """
        if not self.query_engine:
            raise ValueError("Query engine not created. Call create_query_engine() first.")
        
        # Retrieve passages if requested
        retrieved_passages = None
        if retrieve_passages:
            retrieved_passages = self._retrieve_relevant_passages(question)
            
        # Create Aquinas-specific prompt with retrieved passages
        aquinas_prompt = self._create_aquinas_prompt(question, retrieved_passages)
        
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
    
    def _get_source_priority(self, metadata: Dict[str, Any]) -> int:
        """
        Determine the priority level of a source based on its metadata.
        
        Priority levels (lower number = higher priority):
        1 - Aquinas's primary works (Summa Theologiae, Summa Contra Gentiles, etc.)
        2 - Other Aquinas works (Commentaries, Disputed Questions, etc.)
        3 - Secondary sources (commentaries, modern interpretations)
        4 - Unknown or unclear sources
        
        Args:
            metadata: Document metadata
            
        Returns:
            Priority level (1-4)
        """
        file_name = metadata.get('file_name', '').lower()
        source = metadata.get('source', '').lower()
        
        # Primary Aquinas works - highest priority
        primary_works = [
            'summa theologiae', 'summa contra gentiles', 'summa theologica',
            'st', 'scg', 'de veritate', 'de potentia', 'de malo',
            'quodlibetal questions', 'quodlibet', 'disputed questions'
        ]
        
        for work in primary_works:
            if work in file_name or work in source:
                return 1
        
        # Other Aquinas works - high priority
        aquinas_works = [
            'aquinas', 'thomas', 'thomistic', 'commentary', 'commentaries',
            'sentences', 'lombard', 'aristotle', 'augustine'
        ]
        
        for work in aquinas_works:
            if work in file_name or work in source:
                return 2
        
        # Secondary sources - lower priority
        secondary_indicators = [
            'commentary', 'interpretation', 'analysis', 'study', 'introduction',
            'guide', 'handbook', 'modern', 'contemporary', 'scholarly'
        ]
        
        for indicator in secondary_indicators:
            if indicator in file_name or indicator in source:
                return 3
        
        # Unknown sources - lowest priority
        return 4

    def _retrieve_relevant_passages(self, question: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Retrieve relevant passages from the vector store for the given question.
        Prioritizes primary sources over secondary sources.
        
        Args:
            question: The question to retrieve passages for
            top_k: Number of top passages to retrieve
            
        Returns:
            List of relevant passages with scores, prioritized by source type
        """
        if not self.index:
            return []
        
        try:
            # Retrieve more passages initially to allow for prioritization
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k * 2  # Get more to allow for filtering and ranking
            )
            
            # Retrieve relevant nodes
            nodes = retriever.retrieve(question)
            
            # Apply similarity filtering
            similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
            filtered_nodes = similarity_postprocessor.postprocess_nodes(nodes)
            
            # Sort by source priority, then by similarity score
            def sort_key(node_with_score):
                priority = self._get_source_priority(node_with_score.node.metadata)
                score = getattr(node_with_score, 'score', 0.0)
                # Lower priority number = higher priority, so we negate it
                # Higher similarity score = better, so we keep it positive
                return (-priority, score)
            
            # Sort by priority and score
            sorted_nodes = sorted(filtered_nodes, key=sort_key, reverse=True)
            
            # Take only the top_k results
            prioritized_nodes = sorted_nodes[:top_k]
            
            logger.info(f"Retrieved {len(prioritized_nodes)} relevant passages for question: {question}")
            logger.info(f"Source priorities: {[self._get_source_priority(node.node.metadata) for node in prioritized_nodes]}")
            
            return prioritized_nodes
            
        except Exception as e:
            logger.error(f"Error retrieving passages: {e}")
            return []
    
    def get_relevant_passages(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant passages for a question without generating a full response.
        Useful for debugging or analyzing what the system retrieves.
        
        Args:
            question: The question to retrieve passages for
            top_k: Number of top passages to retrieve
            
        Returns:
            List of dictionaries containing passage information
        """
        passages = self._retrieve_relevant_passages(question, top_k)
        
        result = []
        for i, passage in enumerate(passages, 1):
            if hasattr(passage, 'node'):
                node = passage.node
                passage_info = {
                    'rank': i,
                    'text': node.text,
                    'score': getattr(passage, 'score', 0.0),
                    'source': node.metadata.get('source', f'Passage {i}'),
                    'file_name': node.metadata.get('file_name', 'Unknown'),
                    'author': node.metadata.get('author', ''),
                    'title': node.metadata.get('title', ''),
                    'link': node.metadata.get('link', ''),
                    'page_label': node.metadata.get('page_label', ''),
                    'metadata': node.metadata
                }
            else:
                passage_info = {
                    'rank': i,
                    'text': str(passage),
                    'score': 0.0,
                    'source': f'Passage {i}',
                    'file_name': 'Unknown',
                    'metadata': {}
                }
            result.append(passage_info)
        
        return result
        
    def _create_aquinas_prompt(self, question: str, retrieved_passages: list = None) -> str:
        """Create a sophisticated prompt for Aquinas queries with RAG integration."""
        
        # Build context from retrieved passages if available
        context_section = ""
        reference_list = ""
        if retrieved_passages:
            context_section = "\n\nRetrieved Context from Aquinas's Works:\n"
            reference_list = "\n\nReferences:\n"
            
            # Create a list of all citation information for the AI to process
            citation_data = []
            
            for passage in retrieved_passages:
                # Handle NodeWithScore objects
                if hasattr(passage, 'node'):
                    node = passage.node
                    text_content = node.text
                    
                    # Extract source information from metadata
                    author = node.metadata.get('author', '')
                    title = node.metadata.get('title', '')
                    link = node.metadata.get('link', '')
                    source_info = node.metadata.get('source', '')
                    file_name = node.metadata.get('file_name', 'Unknown')
                    page_label = node.metadata.get('page_label', '')
                    leaf_number = node.metadata.get('leaf_number', '')
                    
                    # Create proper citation format with enhanced metadata
                    citation_parts = []
                    
                    # Add author if available
                    if author:
                        citation_parts.append(author)
                    
                    # Add title if available
                    if title:
                        citation_parts.append(f'"{title}"')
                    elif file_name != 'Unknown':
                        citation_parts.append(f'({file_name})')
                    
                    # Add page information if available
                    if page_label:
                        citation_parts.append(f'p. {page_label}')
                    
                    # Add link if available, incorporating leaf number if present
                    if link:
                        # Check if this is an archive.org URL and if we have a leaf number
                        if 'archive.org' in link and leaf_number:
                            # Use leaf_number directly for the URL
                            if 'archive.org/details/' in link:
                                # Insert leaf number before the mode parameter
                                if '/mode/' in link:
                                    link_with_page = link.replace('/mode/', f'/page/n{leaf_number}/mode/')
                                else:
                                    # Add leaf number at the end if no mode parameter
                                    link_with_page = f"{link}/page/n{leaf_number}/mode/2up"
                                citation_parts.append(f'[Link: {link_with_page}]')
                            else:
                                citation_parts.append(f'[Link: {link}]')
                        else:
                            citation_parts.append(f'[Link: {link}]')
                    
                    # Create final citation text
                    if citation_parts:
                        citation_text = ', '.join(citation_parts)
                    else:
                        citation_text = f"{source_info}"
                        if file_name != 'Unknown':
                            citation_text += f" ({file_name})"
                    
                    citation_data.append({
                        'text': text_content,
                        'citation': citation_text,
                        'page_label': page_label,
                        'leaf_number': leaf_number,
                        'file_name': file_name
                    })
                else:
                    # Fallback for other types
                    text_content = str(passage)
                    citation_text = 'Passage'
                    
                    citation_data.append({
                        'text': text_content,
                        'citation': citation_text,
                        'page_label': '',
                        'leaf_number': '',
                        'file_name': 'Unknown'
                    })
            
            # Add all passages to context without manual numbering
            for citation_info in citation_data:
                context_section += f"{citation_info['citation']}:\n> {citation_info['text'].replace(chr(10), chr(10) + '> ')}\n\n"
        
        return f"""You are a specialized scholarly assistant with deep expertise in the complete works of St. Thomas Aquinas (1225-1274). Your responses must be grounded in his actual texts and demonstrate mastery of Thomistic thought.

PRIMARY SOURCE PRIORITY:
- ALWAYS prioritize answering the question using Aquinas's own words from the retrieved primary sources
- When multiple sources are available, give precedence to Aquinas's direct statements
- Distinguish clearly between Aquinas's original texts and later Thomistic tradition
- If secondary sources are present, use them only to supplement, not replace, Aquinas's own words
- Base your analysis primarily on Aquinas's actual texts, not on interpretations or summaries

METHODOLOGY:
- Base your analysis primarily on the retrieved passages provided below
- Use Aquinas's own terminology and conceptual framework
- Distinguish between what Aquinas explicitly states vs. reasonable inferences
- Consider objections and replies in Aquinas's characteristic style
- Show awareness of his sources (especially Aristotle, Augustine, Pseudo-Dionysius)

CITATION REQUIREMENTS:
- Use Wikipedia-style numbered citations throughout your response (e.g., [1], [2], [3])
- Reference specific works using standard abbreviations (ST = Summa Theologiae, SCG = Summa Contra Gentiles, etc.)
- Use precise citations (e.g., ST I, q.2, a.3; SCG II, c.15)
- Distinguish between Aquinas's own position and positions he discusses but rejects
- Note when paraphrasing vs. directly quoting
- Always cite Aquinas's primary texts when available, not secondary sources
- IMPORTANT: You must assign reference numbers [1], [2], [3], etc. based on the ORDER OF APPEARANCE in your response, not the order the sources were retrieved
- The first source you reference in your response should be [1], the second should be [2], and so on
- Include a numbered reference list at the end of your response that corresponds to the citation numbers you used
- For archive.org links, the URL uses leaf numbers with 'n' prefix for navigation (e.g., /page/n162/mode/2up), but reference page labels in your response text

SCHOLARLY STANDARDS:
- Maintain academic precision while being accessible
- Acknowledge limitations or uncertainties in interpretation
- Note when questions go beyond what Aquinas directly addressed
- Distinguish between historical Thomas and later Thomistic tradition when relevant
- When secondary sources are cited, clearly indicate they are secondary and not Aquinas's own words

RESPONSE STRUCTURE:
- As a priority begin the response using Aquinas's own words from retrieved primary sources
- Structure your response to prioritize Aquinas's direct statements
- Use secondary sources only as supplementary support
- Provide comprehensive analysis grounded in Aquinas's actual texts
- Support all claims with appropriate citations from Aquinas's primary sources whenever possible, especially from the retrieved primary sources
- If primary sources are available, they must form the core of your response
- When quoting Aquinas directly, format the quotes using markdown block quotes (> text) for proper visual distinction
- End your response with a "References" section containing numbered citations that correspond to the [1], [2], [3] citations used throughout your response
- The reference numbers should be assigned based on the order you first mention each source in your response, not the retrieval order

{context_section}
QUESTION: {question}

Provide a comprehensive response that demonstrates both textual fidelity to Aquinas and sophisticated theological-philosophical analysis. Structure your response clearly and support all claims with appropriate citations from Aquinas's primary sources whenever possible. Remember to assign reference numbers [1], [2], [3], etc. based on the order you first reference each source in your response, not the order they appear in the retrieved context above."""
        
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
