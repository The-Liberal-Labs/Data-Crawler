# DEPLOYMENT GUIDE

This guide will help you deploy the document library system on a single production server. The system leverages local databases (Milvus, Neo4j, MongoDB), localized document extraction (Docling), and Google's online Gemini APIs (`gemini-2.5-flash-lite`, `gemini-embedding-001`) for heavy AI computing.

## Prerequisites

- **OS**: Linux or macOS recommended
- **Python**: 3.10+
- **Docker**: Docker Engine and Docker Compose (required for Milvus, Neo4j, and MongoDB)
- **Google GenAI API Key**: Obtain from [Google AI Studio](https://aistudio.google.com/)

## Deployment Steps

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd document-library
```

### 2. Configure the Environment

The system relies on a unified configuration.

```bash
# Copy the environment template
cp .env.template .env
```

Open `.env` in your preferred editor and set the required variables:

```env
# Security
SECRET_KEY=your_secure_random_string_here

# Google GenAI Settings
GEMINI_API_KEY=your_actual_gemini_api_key

# Database Settings
MILVUS_HOST=localhost
MILVUS_PORT=19530

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_neo4j_password

MONGODB_URI=mongodb://localhost:27017
```

### 3. Deploy Databases via Docker

We use Docker to spin up the local databases securely.

```bash
# Start MongoDB, Neo4j, and Milvus in the background
docker-compose -f docker-compose.databases.yml up -d

# Verify they are running
docker-compose -f docker-compose.databases.yml ps
```

*Note: Milvus relies on its internal components (etcd, minio). Ensure your Docker deployment has sufficient memory allocation.*

### 4. Setup Python Environment & Install Dependencies

```bash
# Create and activate a Virtual Environment
python -m venv venv
source venv/bin/activate

# Install all backend, router, and extraction dependencies
# This includes the new `google-genai` SDK
pip install -r requirements.txt
```

### 5. Start Backend Services

The backend consists of the Orchestrator API, Docling extraction service, and the Gemini API routing microservices.

```bash
# Start all services
./start_services.sh
```

Alternately, if you wish to run them manually in separate terminal windows:

```bash
# Terminal 1: Orchestrator
source venv/bin/activate
cd orchestrator_api
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Docling Extraction Service
source venv/bin/activate
cd gpu_services/docling_service
python main.py

# Terminal 3: LLM Service (Gemini Router)
source venv/bin/activate
cd gpu_services/llm_service
python main.py

# Terminal 4: Embedding & Knowledge Graph Services
source venv/bin/activate
cd gpu_services/embedding_service && python main.py &
cd gpu_services/knowledge_graph_service && python main.py &
```

## Service Health Checks

After deployment, verify all services are running and accessible:

- **Orchestrator API**: `curl http://localhost:8000/health`
- **Docling Service**: `curl http://localhost:8004/health`
- **LLM Service**: `curl http://localhost:8001/health`
- **Embedding Service**: `curl http://localhost:8002/health`
- **Knowledge Graph Service**: `curl http://localhost:8003/health`

## Production Considerations

1. **Security & API Keys**:
   - Never expose your `GEMINI_API_KEY` to the public frontend. All Gemini interactions route securely through the python `google-genai` client inside your `llm_service` and `embedding_service`.
   - Update default MongoDB and Neo4j credentials.
   - Use TLS/HTTPS in front of the Orchestrator API (e.g., using Nginx or Traefik).

2. **Database Persistence**:
   - Ensure the Docker volumes for Milvus, MongoDB, and Neo4j are properly mapped to your host machine's disk so data is not lost when containers restart. (Configured in the `docker-compose.databases.yml`).

3. **Monitoring**:
   - Track logs using `docker logs -f milvus` (or database of choice).
   - Use standard OS monitoring tools (htop) to track RAM and CPU usage, as Docling can be resource-intensive during document ingestion.

## Support

If you encounter issues:
1. Verify the Google GenAI API key is valid.
2. Ensure Docker containers are fully healthy.
3. Check the Python logs for any syntax or dependency missing errors.