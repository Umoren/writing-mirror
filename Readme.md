# Writing Mirror

A personal writing assistant that learns from your existing content to generate suggestions that preserve your unique writing style and voice.

## 🎯 What It Does

Instead of generic AI suggestions, this tool analyzes your previous writing (from Notion) to understand your style, tone, and patterns. When you're writing, it suggests continuations that sound like *you* wrote them.

**Key Features:**
- 📝 Learns from your Notion documents
- 🎨 Preserves your unique writing style  
- ⚡ Fast semantic search using vector embeddings
- 🔒 Self-hosted - your data stays with you
- 🌐 Chrome extension for seamless Notion integration

## 🏗️ Architecture

This project replaces the original Ragie-based implementation with a self-hosted solution:

```
Notion Documents → Document Processing → Embeddings → Vector Database → API → Chrome Extension
```

**Tech Stack:**
- **Backend**: FastAPI + Python
- **Vector Database**: Qdrant  
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Caching**: Redis
- **Frontend**: Chrome Extension
- **Data Source**: Notion API

## 📁 Project Structure

```
voice-writing-assistant/
├── app/                          # Main application
│   ├── api/                      # API endpoints
│   ├── models/                   # Pydantic models
│   ├── services/                 # Core business logic
│   │   ├── notion_service.py     # Notion API integration
│   │   ├── document_processor.py # Text chunking and processing
│   │   ├── embedding_service.py  # Vector embeddings
│   │   ├── vector_service.py     # Qdrant integration
│   │   └── integration_service.py # Orchestration service
│   └── utils/                    # Utilities and config
├── chrome-extension/             # Chrome extension (WIP)
├── scripts/                      # Test and utility scripts
├── data/                         # Data storage and state
└── tests/                        # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for Qdrant and Redis)
- Notion API access
- Chrome browser (for extension)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/voice-writing-assistant.git
cd voice-writing-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Services

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Redis  
docker run -p 6379:6379 redis:alpine
```

### 3. Configuration

Create a `.env` file:

```env
# Notion API
NOTION_API_KEY=your_notion_integration_key
NOTION_DATABASE_ID=your_notion_database_id

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=writing_samples

# Redis
REDIS_URL=redis://localhost:6379/0

# Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Sync Your Documents

```bash
# Test the integration
python scripts/test_integration.py

# This will:
# 1. Connect to your Notion database
# 2. Process documents into chunks
# 3. Generate embeddings  
# 4. Store in Qdrant vector database
```

### 5. Start the API

```bash
# Run the FastAPI server
uvicorn app.main:app --reload --port 8001

# API will be available at http://localhost:8001
# Docs at http://localhost:8001/docs
```

## 📊 Current Status

### ✅ Implemented
- **Core Services**: All backend services working
- **Notion Integration**: Successfully syncing documents
- **Vector Database**: Qdrant integration with embeddings
- **Document Processing**: Chunking and embedding generation
- **Search Functionality**: Semantic search working

### 🚧 In Progress  
- **API Endpoints**: REST API for suggestions
- **Performance Optimization**: Caching layer

### ❌ Todo
- **Chrome Extension**: Frontend integration
- **LLM Integration**: Better suggestion generation
- **Advanced Chunking**: Smarter document segmentation
- **Incremental Sync**: Real-time document updates

## 🧪 Testing

```bash
# Test individual components
python scripts/test_notion_service.py      # Test Notion connection
python scripts/test_vector_service.py      # Test Qdrant operations  
python scripts/test_embedding.py           # Test embedding generation
python scripts/test_integration.py         # Test full pipeline

# Run test suite
pytest tests/
```

## 📈 Performance Metrics

Current benchmarks (28 documents → 993 chunks):

- **Document Processing**: ~4 minutes for initial sync
- **Embedding Generation**: ~2 minutes for 993 chunks  
- **Vector Search**: <100ms for similarity queries
- **Memory Usage**: ~500MB for loaded models

## 🔧 Configuration Options

### Document Processing
```python
# app/services/document_processor.py
chunk_size = 512          # Characters per chunk
chunk_overlap = 128       # Overlap between chunks
min_chunk_size = 50       # Minimum viable chunk size
```

### Vector Search
```python
# Default search parameters
top_k = 5                 # Number of similar chunks to retrieve
score_threshold = 0.7     # Minimum similarity score
```

## 🤝 API Usage

### Generate Suggestions

```bash
curl -X POST "http://localhost:8001/api/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I was thinking about machine learning and how it",
    "context": "Technical blog post about AI",
    "task": "continue",
    "num_suggestions": 3
  }'
```

### Response Format

```json
{
  "trace_id": "uuid-here",
  "suggestions": [
    {
      "text": "impacts our daily workflows and productivity",
      "score": 0.85
    }
  ],
  "sources": [
    {
      "doc_id": "notion-page-id", 
      "title": "ML in Practice",
      "similarity": 0.85
    }
  ],
  "stats": {
    "total_time_ms": 150,
    "search_time_ms": 50,
    "generation_time_ms": 100
  }
}
```

## 🛣️ Roadmap

### Phase 1: Core Functionality (Current)
- [x] Document syncing from Notion
- [x] Vector embeddings and search
- [ ] REST API endpoints
- [ ] Basic caching

### Phase 2: User Experience  
- [ ] Chrome extension
- [ ] Real-time suggestions
- [ ] User feedback collection
- [ ] Performance optimization

### Phase 3: Advanced Features
- [ ] Multiple data sources (Google Docs, Markdown files)
- [ ] Advanced LLM integration
- [ ] Style analysis and metrics
- [ ] Collaborative features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with sentence-transformers for embeddings
- Uses Qdrant for vector similarity search
- Notion API for document management

---

**Note**: This is an active development project. Features and documentation are continuously being updated.