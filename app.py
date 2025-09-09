"""
FastAPI Application for Aquinas RAG System
==========================================

This FastAPI application provides endpoints for querying and uploading documents
to the sophisticated Aquinas RAG system using LlamaIndex.
"""

import os
import sys
import logging
import tempfile
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import the Aquinas RAG system
from aquinas_rag import AquinasRAGSystem, AquinasDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: Optional[AquinasRAGSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - initialize and cleanup resources."""
    global rag_system
    
    # Initialize the RAG system on startup
    try:
        logger.info("Initializing Aquinas RAG system with Pinecone + OpenAI + LlamaCloud...")
        rag_system = AquinasRAGSystem(
            llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY")
        )
        # Ensure the index wrapper and query engine are ready on startup
        try:
            rag_system.ensure_ready_for_queries()
            logger.info("RAG system query engine ready on startup")
        except Exception as e:
            logger.warning(f"RAG system initialized, but query engine not ready yet: {e}")
        logger.info("Aquinas RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Aquinas RAG system...")

# Create FastAPI application
app = FastAPI(
    title="Aquinas RAG API",
    description="Sophisticated RAG system for St. Thomas Aquinas works using Pinecone, OpenAI, and LlamaCloud",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="The question to ask about Aquinas", min_length=1)
    context_length: int = Field(4000, description="Maximum context length", ge=1000, le=8000)

class QueryResponse(BaseModel):
    """Response model for query results."""
    answer: str = Field(..., description="The generated answer")
    context_length: int = Field(..., description="Context length used")

class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str = Field(..., description="Upload status message")
    documents_processed: int = Field(..., description="Number of documents processed")
    file_names: List[str] = Field(..., description="Names of processed files")
    index_status: str = Field(..., description="Status of the index after upload")
    total_documents: int = Field(..., description="Total documents in the index")

class StatusResponse(BaseModel):
    """Response model for system status."""
    rag_system_initialized: bool = Field(..., description="Whether RAG system is initialized")
    index_exists: bool = Field(..., description="Whether vector index exists")
    query_engine_ready: bool = Field(..., description="Whether query engine is ready")
    status_message: str = Field(..., description="Overall system status message")


async def process_uploaded_file(file: UploadFile) -> List[Any]:
    """Process an uploaded file and return documents."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Process the document
        documents = rag_system.ingest_documents(
            documents_path=temp_file_path
        )
        return documents
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Aquinas RAG API",
        "description": "Sophisticated RAG system for St. Thomas Aquinas works using Pinecone, OpenAI, and LlamaCloud",
        "version": "1.0.0",
        "configuration": {
            "vector_store": "Pinecone",
            "llm_provider": "OpenAI GPT-4o",
            "embedding_provider": "OpenAI text-embedding-3-large",
            "parsing": "LlamaCloud"
        },
        "endpoints": {
            "query": "/query",
            "upload": "/upload",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_aquinas(request: QueryRequest):
    """
    Query the Aquinas RAG system with a question.
    
    This endpoint allows you to ask questions about St. Thomas Aquinas's works
    and get comprehensive answers based on the indexed documents.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.query_engine:
        try:
            rag_system.ensure_ready_for_queries()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Query engine not ready: {str(e)}")
    
    try:
        
        # Query the RAG system
        answer = rag_system.query(
            question=request.query,
            context_length=request.context_length
        )
        print(answer) 
        return QueryResponse(
            answer=answer,
            context_length=request.context_length
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to upload"),
):
    """
    Upload a PDF document to the Aquinas RAG system.
    
    This endpoint processes PDF documents using LlamaCloud parsing and adds them 
    to the Pinecone vector index for querying. Documents are chunked using 
    advanced strategies optimized for Aquinas's philosophical texts.
    """
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
    except HTTPException:
        raise
    
    try:
        # Process the uploaded file
        documents = await process_uploaded_file(file)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed from the file")
        
        # Build or update the index
        if rag_system.index is None:
            # First document - build new index
            logger.info("Building new index with first document...")
            rag_system.build_index(documents)
            rag_system.create_query_engine()
            index_status = "New index created"
        else:
            # Additional documents - add to existing index
            logger.info("Adding document to existing index...")
            rag_system.add_documents_to_index(documents)
            index_status = "Document added to existing index"
        
        # Get total document count (approximate)
        total_documents = len(documents) if rag_system.index is None else len(documents) + 1
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            documents_processed=len(documents),
            file_names=[file.filename],
            index_status=index_status,
            total_documents=total_documents
        )
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_system_status():
    """
    Get the current status of the RAG system.
    
    This endpoint provides information about whether the system is ready
    to process queries and the state of the vector index.
    """
    if rag_system is None:
        return StatusResponse(
            rag_system_initialized=False,
            index_exists=False,
            query_engine_ready=False,
            status_message="RAG system not initialized"
        )
    
    # Try to ensure readiness if not already ready
    if rag_system.index is None or rag_system.query_engine is None:
        try:
            rag_system.ensure_ready_for_queries()
        except Exception:
            pass
    index_exists = rag_system.index is not None
    query_engine_ready = rag_system.query_engine is not None
    
    if not index_exists:
        status_message = "RAG system initialized but no documents uploaded yet"
    elif not query_engine_ready:
        status_message = "Index exists but query engine not ready"
    else:
        status_message = "System ready for queries"
    
    return StatusResponse(
        rag_system_initialized=True,
        index_exists=index_exists,
        query_engine_ready=query_engine_ready,
        status_message=status_message
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('aquinas_rag.log')
        ]
    )

def check_environment():
    """Check if required environment variables are set."""
    required_vars = []
    
    # Check for at least one LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    
    if llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            required_vars.append("OPENAI_API_KEY")
    elif llm_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            required_vars.append("ANTHROPIC_API_KEY")
    elif llm_provider == "ollama":
        # Ollama doesn't require API key, but we should check if it's running
        pass
    
    # Check embedding provider (currently only OpenAI supported)
    if not os.getenv("OPENAI_API_KEY"):
        required_vars.append("OPENAI_API_KEY (for embeddings)")
    
    if required_vars:
        print("❌ Missing required environment variables:")
        for var in required_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("See env.example for reference.")
        return False
    
    return True

def main():
    """Main function to start the Aquinas RAG API server."""
    parser = argparse.ArgumentParser(
        description="Aquinas RAG API - Sophisticated RAG system for St. Thomas Aquinas works"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Print startup information
    print("🏛️  Aquinas RAG API")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Log Level: {args.log_level}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print("=" * 50)
    
    # Configuration summary
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    vector_store = os.getenv("VECTOR_STORE", "chroma")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    
    print(f"LLM Provider: {llm_provider}")
    print(f"Embedding Provider: {embedding_provider}")
    print(f"Vector Store: {vector_store}")
    print("=" * 50)
    
    # Start the server
    try:
        logger.info("Starting Aquinas RAG API server...")
        uvicorn.run(
            "app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
