# MediMind AI Architecture

MediMind AI is split into a Django application layer and a FastAPI AI service.

## Django

Django owns authentication, user medical profiles, uploaded report records, health-score history, timelines, predictions, recommendations, and knowledge-document metadata. It calls FastAPI for report NLP, ML prediction, health-score recalculation, and knowledge-base indexing.

## FastAPI

FastAPI owns model loading, prediction, SHAP explanations, LLM provider routing, report text extraction, RAG indexing, RAG retrieval, multi-agent coordination, and Prometheus metrics.

## Data Stores

- PostgreSQL stores application data and MLflow metadata.
- Qdrant stores 1024-dimensional Cohere embeddings.
- MLflow artifacts store model-training outputs and inference logs.

## Safety Flow

Emergency keyword detection runs before intent classification. Medication responses forbid prescriptions and dosage suggestions. Diagnostic responses are educational and require provider consultation.
