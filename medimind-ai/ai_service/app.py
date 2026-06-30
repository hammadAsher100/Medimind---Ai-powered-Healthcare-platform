from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from model_registry import load_all_models
from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.vector_store.qdrant_store import QdrantStore
from routers import agents, comparison, health_score, knowledge_base, prediction, report_analyzer

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_all_models()
    if os.environ.get("DISABLE_QDRANT", "False").lower() == "true":
        app.state.qdrant_store = None
    else:
        try:
            app.state.qdrant_store = QdrantStore()
            app.state.qdrant_store.create_collection()
        except Exception:
            app.state.qdrant_store = None
    app.state.cohere_embedder = CohereEmbedder()
    yield


app = FastAPI(title="MediMind AI Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)
app.include_router(report_analyzer.router)
app.include_router(knowledge_base.router)
app.include_router(health_score.router)
app.include_router(comparison.router)
app.include_router(agents.router)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "medimind-ai"}
