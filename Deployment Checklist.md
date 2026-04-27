# 📋 Deployment Checklist

Use this checklist to ensure your document library system is properly deployed and configured on your unified production server.

## Pre-Deployment Preparation

### Hardware Prerequisites
- [ ] OS: Linux or macOS (capable of background processing)
- [ ] Python 3.10+ installed
- [ ] Docker & Docker Compose installed
- [ ] At least 50GB free storage for DB volumes
- [ ] Minimum 16GB RAM (32GB recommended for Docling extractions and Milvus memory requirements)

### API & Service Prerequisites
- [ ] Obtain a Google Gemini API Key from Google AI Studio.

## Deployment Steps

### Step 1: Clone and Configure Environment
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Copy environment template: `cp .env.template .env`
- [ ] Edit `.env` with your configuration:
  - [ ] Set `SECRET_KEY` to a secure random value
  - [ ] Set `GEMINI_API_KEY` to your valid Google Gemini token
  - [ ] Configure `NEO4J_PASSWORD`
  - [ ] Ensure `MILVUS_HOST=localhost` and ports are correct

### Step 2: Database Initialization (Docker)
- [ ] Run: `docker-compose -f docker-compose.databases.yml up -d`
- [ ] Verify Milvus is healthy: `docker ps | grep milvus`
- [ ] Verify Neo4j is healthy: Check `http://localhost:7474`
- [ ] Verify MongoDB is healthy: Expected on port `27017`

### Step 3: Application Setup
- [ ] Create Python virtual environment: `python -m venv venv`
- [ ] Activate environment: `source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify `google-genai` SDK is properly installed

### Step 4: Run Services
- [ ] Start backend and proxy services: `./start_services.sh`
- [ ] Verify Orchestrator API is running: `curl http://localhost:8000/health`
- [ ] Verify LLM/Gemini Proxy is running: `curl http://localhost:8001/health`
- [ ] Verify Embeddings Proxy is running: `curl http://localhost:8002/health`
- [ ] Verify Graph Extraction Service is running: `curl http://localhost:8003/health`
- [ ] Verify Docling Service is running: `curl http://localhost:8004/health`

## Post-Deployment Verification

### System Health Checks
- [ ] Run comprehensive test: `./test_system.sh`
- [ ] All services return HTTP 200 on health endpoints.
- [ ] No immediate connection refusal errors in logs.

### Functional Testing
- [ ] User registration works: Test signup endpoint
- [ ] Authentication works: Test login and get JWT token
- [ ] Document upload works: Upload a test PDF/DOCX
- [ ] Document processing completes: Extracted cleanly by local Docling
- [ ] Vectorization completes: Text sent to `gemini-embedding-001` and saved to Milvus
- [ ] Graph Extraction completes: Relationships extracted using `gemini-2.5-flash-lite` and cypher transactions executed on Neo4j.
- [ ] Query system responds: Test basic query functionality combining vector retrieval and graph RAG.

## Production Readiness

### Security
- [ ] Change default passwords for Neo4j and MongoDB.
- [ ] Generate secure JWT secret key (min 32 characters).
- [ ] Confirmed `GEMINI_API_KEY` is completely hidden from the client browser and only accessed via the Python backend.

### Backup Strategy
- [ ] MongoDB backup strategy configured
- [ ] Neo4j backup script configured  
- [ ] Milvus backups configured or data volumes preserved

### Monitoring
- [ ] Server resources monitored (CPU, Memory limits).

## Common Issues Checklist

If you encounter problems, check these items:

### Service Won't Start
- [ ] Check Python version (must be 3.10+)
- [ ] Verify virtual environment activated before starting scripts
- [ ] Check all dependencies installed 
- [ ] Check port availability (`netstat -tulpn | grep PORT`)

### API Failures (Google Gemini)
- [ ] Double-check `GEMINI_API_KEY` validity.
- [ ] Check standard output logs for `google.genai.errors.APIError` or rate-limiting responses.

### Database Connection Issues
- [ ] Check database Docker containers are running (`docker ps`)
- [ ] Verify connection strings in `.env`
- [ ] Check Neo4j Cypher logs if Graph relations are failing to save.

---

✅ **System deployment complete when all items are checked!**