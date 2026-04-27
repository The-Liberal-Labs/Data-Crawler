# ⚡ Quick Reference

## 🎯 Key System Updates
1. **Single Machine Deployment**: Replaced complex multi-device network bridging. All services run uniformly on a single server structure.
2. **AI Models Migration**: 
   - Moved from local Gemma 3-4B to **Google `gemini-2.5-flash-lite`** (via `google-genai` SDK).
   - Moved from local EmbeddingGemma to **Google `gemini-embedding-001`**.
3. **Local Processing**: Kept **Docling** for private, robust document extraction (PDF, DOCX).
4. **Data Stores**: **Milvus** (Vector) and **Neo4j** (Graph using powerful Cypher queries) remain local and connected via Docker.

## 🚀 Quick Deployment Commands

```bash
# 1. Environment configuration
cp .env.template .env
# Edit .env to add your GEMINI_API_KEY and database credentials

# 2. Start Databases
docker-compose -f docker-compose.databases.yml up -d

# 3. Setup Dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Start Services
./start_services.sh
```

## 📊 Service Endpoints & Architecture Map

| Service | Address | Purpose |
|---------|---------|---------|
| **Orchestrator** | `http://localhost:8000` | Main Application API |
| **Docling** | `http://localhost:8004` | PDF/Data local processor |
| **LLM Proxy** | `http://localhost:8001` | Routes requests to `gemini-2.5-flash-lite` |
| **Embedding Proxy** | `http://localhost:8002` | Routes strings to `gemini-embedding-001` |
| **Knowledge Graph** | `http://localhost:8003` | Builds cypher transactions |
| **Neo4j (Database)** | `bolt://localhost:7687` | Knowledge graph via Cypher |
| **Neo4j (UI)** | `http://localhost:7474` | Cypher Playground & Visualizer |
| **Milvus**| `localhost:19530` | Vector / Semantic Search |
| **MongoDB** | `mongodb://localhost:27017` | User & State storage |

## 🔍 Enhanced Data Flow

```text
Document Upload
      ↓
Docling Local Processing (Text/Table structured extraction)
      ↓
Entity & Relation Extraction -> Powered by Gemini 2.5 Flash Lite
      ↓  
Neo4j Graph Storage -> via Cypher MERGE queries
      ↓
Embedding Generation -> Powered by Gemini Embedding 001
      ↓
Milvus Storage -> Fast HNSW vector index
```

## 🔧 Important Configuration Notes

### `.env` File Updates:
```env
# Required for GenAI features
GEMINI_API_KEY=your_google_ai_studio_api_key

# Proper local database routing
NEO4J_URI=bolt://localhost:7687
MILVUS_HOST=localhost
```

## ⚠️ Critical Notes
- **Cypher Queries**: Ensure that any manual updates to graph logic utilize standard Cypher Language (e.g., `MATCH`, `MERGE`, `CREATE`). Avoid plain text injection.
- **Milvus Lifecycle**: Milvus can consume heavy RAM. Make sure Docker is allocated enough resource room.
- **Docling Extraction**: Time to process a document primarily depends on your local machine's speed to run Docling's local MLX framework.

## 🐛 Quick Debugging

```bash
# Check Docker containers
docker ps

# Test Gemini reachability
python -c "from google import genai; import os; c=genai.Client(api_key=os.environ.get('GEMINI_API_KEY')); print(c.models.generate_content(model='gemini-2.5-flash-lite', contents='Test').text)"

# Check service health individually
curl http://localhost:8000/health
curl http://localhost:8001/health
```