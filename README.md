# Aquinas RAG API

A sophisticated FastAPI-based RAG (Retrieval-Augmented Generation) system specifically designed for analyzing and querying the works of St. Thomas Aquinas using LlamaIndex.

## Features

- **Advanced Document Processing**: Sophisticated parsing and chunking strategies optimized for Aquinas's philosophical texts
- **Multiple LLM Support**: OpenAI, Anthropic, and Ollama integration
- **Vector Database Integration**: ChromaDB, Pinecone, and Weaviate support
- **Work Type Classification**: Automatic detection and filtering by Aquinas work types
- **RESTful API**: Clean FastAPI endpoints for querying and document upload
- **Advanced Chunking**: Semantic, hierarchical, and enhanced sentence splitting strategies

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd lumen-api

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your API keys
nano .env
```

**Required Environment Variables:**

- `OPENAI_API_KEY`: Your OpenAI API key (required for embeddings and optional for LLM)
- At least one LLM provider key:
  - `ANTHROPIC_API_KEY` (for Anthropic Claude)
  - `OPENAI_API_KEY` (for OpenAI GPT)
  - Ollama (no API key needed, runs locally)

### 3. Run the Server

```bash
# Start the development server
python main.py

# Or with custom options
python main.py --host 0.0.0.0 --port 8080 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check and system status
- `GET /work-types` - Available Aquinas work types
- `POST /query` - Query the RAG system
- `POST /upload` - Upload PDF documents
- `GET /metadata` - Get indexed document metadata
- `GET /chunking-info` - Get chunking strategy information

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Usage Examples

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@summa_theologiae.pdf" \
  -F "work_type=Summa Theologiae"
```

### 2. Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does Aquinas say about the existence of God?",
    "work_type_filter": "Summa Theologiae",
    "context_length": 4000
  }'
```

### 3. Python Client Example

```python
import requests

# Upload a document
with open("aquinas_work.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f},
        data={"work_type": "Summa Theologiae"}
    )
print(response.json())

# Query the system
query_response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Explain Aquinas's five ways to prove God's existence",
        "work_type_filter": "Summa Theologiae"
    }
)
print(query_response.json()["answer"])
```

## Configuration

### LLM Providers

**OpenAI (Recommended)**

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

**Anthropic Claude**

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Ollama (Local)**

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Vector Stores

**ChromaDB (Local Development)**

```env
VECTOR_STORE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

**Pinecone (Production)**

```env
VECTOR_STORE=pinecone
PINECONE_API_KEY=your-pinecone-key
```

**Weaviate**

```env
VECTOR_STORE=weaviate
WEAVIATE_URL=your-weaviate-url
WEAVIATE_API_KEY=your-weaviate-key
```

## Aquinas Work Types

The system recognizes and can filter by these work types:

- **Summa Theologiae**: His most famous systematic theology
- **Summa Contra Gentiles**: Apologetic work against non-Christians
- **Commentaries**: Commentaries on Aristotle and other philosophers
- **Disputed Questions**: Academic disputations on various topics
- **Treatises**: Various theological and philosophical treatises
- **Sermons**: Sermons and homilies
- **Letters**: Correspondence and letters
- **Prayers**: Prayers and devotions
- **Other**: Other works not specifically categorized

## Advanced Features

### Chunking Strategies

The system uses sophisticated chunking strategies optimized for Aquinas's texts:

- **Semantic Chunking**: For argument-based works (Commentaries, Disputed Questions)
- **Hierarchical Chunking**: For structured works (Summa Theologiae)
- **Enhanced Sentence Splitting**: Fallback with improved overlap

### Metadata Extraction

The system automatically extracts:

- Work type classification
- Part, Question, and Article references (for Summa Theologiae)
- File metadata and source information

## Development

### Running in Development Mode

```bash
# Start with auto-reload
python main.py --reload

# Start with debug logging
python main.py --log-level DEBUG
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests (when available)
pytest
```

## Production Deployment

### Using Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn

```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **ChromaDB Issues**: Check if the persist directory is writable
3. **Memory Issues**: Reduce chunk size or use a smaller model
4. **PDF Parsing Issues**: Consider using LlamaCloud API for better parsing

### Logs

Check the application logs:

```bash
tail -f aquinas_rag.log
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [LlamaIndex](https://www.llamaindex.ai/)
- Inspired by the works of St. Thomas Aquinas
