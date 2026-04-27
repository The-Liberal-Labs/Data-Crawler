# Document Library System

A high-performance, scalable document intelligence system running on a single production machine, utilizing Google's advanced Gemini AI models alongside local document processing, graph, and vector databases for a complete, robust architecture.

## 🏗️ System Architecture

### Production Deployment (Single Machine)
The system is designed to run seamlessly on a single unified production server or local machine.

- **Main Backend / Orchestrator API**: Manages business logic, JWT authentication, and file routing.
- **Document Processing (Docling)**: Local service for secure, robust document extraction (text, tables, images).
- **LLM Service (Google Gemini)**: Intelligent router and processing using `gemini-2.5-flash-lite` via the Google GenAI SDK.
- **Embedding Service (Google Gemini)**: High-quality vectorization using `gemini-embedding-001` via the Google GenAI SDK.
- **Graph Database (Neo4j)**: Advanced knowledge graph storage using Cypher.
- **Vector Database (Milvus)**: High-performance HNSW indexing for semantic search.
- **Document Store (MongoDB)**: Scalable storage for user metadata and document processing state.

### Key Features
- **Hybrid AI Approach**: Leverages powerful online Google Gemini models for LLM and embeddings, while keeping document extraction (Docling) and storage local.
- **Multi-modal Processing**: Handles PDF, DOCX, PPTX, HTML, CSV, Excel, images, and audio.
- **Advanced Knowledge Graphs**: Automatic entity extraction and relationship mapping using Neo4j and precise Cypher queries.
- **Semantic Search**: High-performance vector search in Milvus v2.6+.
- **JWT Authentication**: Secure user management with proper session handling.
- **Real-time Processing**: Background document processing with status tracking.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker and Docker Compose (Required for Milvus and Neo4j)
- Google Gemini API Key (Get one from [Google AI Studio](https://aistudio.google.com/))

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd document-library
```

### 2. Configure Environment

1. Copy the environment template:
```bash
cp .env.template .env
```
2. Edit `.env` to include your API Keys and database credentials:
```env
# Google GenAI Settings
GEMINI_API_KEY="your-gemini-api-key-here"

# Database Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-secure-password
MILVUS_HOST=localhost
```

### 3. Deploy the System

We provide a streamlined deployment process:

```bash
# Start Databases (Milvus, Neo4j, MongoDB)
docker-compose -f docker-compose.databases.yml up -d

# Install Python dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start all Backend Services
./start_services.sh
```

### 4. Test the System

```bash
chmod +x test_system.sh
./test_system.sh
```

## 📊 Service Endpoints

All services are exposed locally for security and ease of access:

- **Main API**: `http://localhost:8000`
- **Main API Docs**: `http://localhost:8000/docs`
- **Docling Service**: `http://localhost:8004`
- **LLM Service (Gemini Router)**: `http://localhost:8001`
- **Embedding Service (Gemini Router)**: `http://localhost:8002`
- **Knowledge Graph Service**: `http://localhost:8003`
- **Neo4j Browser**: `http://localhost:7474`
- **Milvus / Vector DB**: `localhost:19530`
- **MongoDB**: `mongodb://localhost:27017`

## 🔧 API Usage Examples

### 1. User Registration
```bash
curl -X POST "http://localhost:8000/api/v1/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "securepass123"}'
```

### 2. Login
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepass123"
```

### 3. Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf"
```

### 4. Check Processing Status
```bash
curl "http://localhost:8000/api/v1/documents/DOC_ID/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 5. Query Documents
```bash
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What are the main topics in my documents?"}'
```

## 🔍 Processing Pipeline

1. **Document Upload**: User uploads file through the orchestrator API.
2. **Content Extraction**: Local Docling service extracts clean text, images, and tables.
3. **Chunking**: Content is intelligently chunked with context preservation.
4. **Entity Extraction**: `gemini-2.5-flash-lite` dynamically extracts entities and maps relationships.
5. **Knowledge Graph**: Extracted entities are stored in Neo4j using robust Cypher transactions.
6. **Embeddings**: Text chunks are vectorized using `gemini-embedding-001`.
7. **Vector Storage**: Embeddings are indexed in Milvus for blazing-fast semantic retrieval.
8. **Query Processing**: Vector Search + Graph RAG combined by Gemini to provide comprehensive answers.

## 🕸️ Neo4j & Cypher Query Language Guide

The system uses advanced Knowledge Graphs to improve referencing and document context. We interact with Neo4j through the standard **Cypher query language**.

If you are developing or reading the graphs directly (via the Neo4j Browser at `http://localhost:7474`), here are some essential Cypher patterns we use:

### 1. Retrieving a Document's Entities
```cypher
MATCH (d:Document {id: $doc_id})-[:CONTAINS]->(e:Entity)
RETURN e.name, e.type, e.description
```

### 2. Finding Relationships Between Entities
```cypher
MATCH (e1:Entity)-[r]->(e2:Entity)
WHERE e1.name CONTAINS 'AI' OR e2.name CONTAINS 'AI'
RETURN e1.name, type(r), e2.name
LIMIT 50
```

### 3. Proper Data Insertion (MERGE vs CREATE)
To ensure we don't create duplicate entities, the Graph builder service uses `MERGE`:
```cypher
MERGE (e1:Entity {name: $entity_name})
SET e1.type = $entity_type, e1.updated_at = timestamp()
MERGE (e2:Entity {name: $target_name})
MERGE (e1)-[r:RELATES_TO {type: $relation_type}]->(e2)
```
*Note: Always use parameters (`$param_name`) in application code to prevent Cypher injection.*

## 🛠️ Technology Stack

- **LLM**: Google `gemini-2.5-flash-lite` (via google-genai SDK)
- **Embeddings**: Google `gemini-embedding-001` (via google-genai SDK)
- **Document Processing**: Local Docling
- **Databases**: Milvus 2.6+ (Vector), Neo4j (Graph), MongoDB (Document)
- **Backend Framework**: FastAPI (Python)

## 🐛 Troubleshooting

### Common Issues

1. **Missing Google API Key**: 
   Ensure `GEMINI_API_KEY` is present in your `.env`. If using the new SDK, verify with:
   ```python
   from google import genai
   import os
   client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
   ```

2. **Neo4j Cypher Execution Errors**:
   If queries fail, check the Neo4j logs. Ensure you aren't passing naked strings where property dictionaries are expected. Use the Neo4j Desktop browser (`localhost:7474`) to manually test Cypher syntax.

3. **Milvus Connection Failed**:
   Ensure Docker containers are running. `docker-compose -f docker-compose.databases.yml ps`

### Log Locations
- **System Services (FastAPI)**: Console output or respective service `logs/` directory.
- **Databases**: `docker logs milvus` or `docker logs neo4j`.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.