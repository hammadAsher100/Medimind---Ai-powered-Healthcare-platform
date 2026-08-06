from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.vector_store.qdrant_store import QdrantStore
from cnn.registry import CNNModelRegistry
from routers import agents, cnn_inference, comparison, health_score, knowledge_base, prediction, report_analyzer
from routers import (
    lab_intelligence,
    patient_state,
    medication_safety,
    counterfactual,
    conflict_detection,
    clinician_review,
    fhir_export,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.cnn_registry = CNNModelRegistry()
    if os.environ.get("DISABLE_CNN", "False").lower() == "true":
        pass
    else:
        app.state.cnn_registry.load_all()
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
cors_origins_str = os.environ.get("FASTAPI_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:18000,http://django:8000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins_str.split(",") if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["accept", "authorization", "content-type", "x-csrftoken", "x-requested-with"],
)

app.include_router(prediction.router)
app.include_router(report_analyzer.router)
app.include_router(knowledge_base.router)
app.include_router(health_score.router)
app.include_router(comparison.router)
app.include_router(agents.router)
app.include_router(cnn_inference.router)
# Clinical intelligence routers
app.include_router(lab_intelligence.router)
app.include_router(patient_state.router)
app.include_router(medication_safety.router)
app.include_router(counterfactual.router)
app.include_router(conflict_detection.router)
app.include_router(clinician_review.router)
app.include_router(fhir_export.router)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "medimind-ai",
        "cnn_models": getattr(app.state, "cnn_registry", CNNModelRegistry()).status(),
    }
