# MediMind AI

MediMind AI is a production-oriented multi-agent healthcare platform with a Django REST backend, a FastAPI AI service, PostgreSQL, Qdrant, MLflow, Prometheus, Grafana, and Nginx.

## Services

- Django API: `http://localhost:8000`
- FastAPI AI service: `http://localhost:8001`
- Nginx gateway: `http://localhost`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Qdrant: `http://localhost:6333`

## Setup

1. Copy `.env` values and replace placeholder secrets and API keys.
2. Start all services:

```bash
docker compose up --build
```

3. Create a Django superuser:

```bash
docker compose exec django python manage.py createsuperuser
```

4. Place disease datasets in `datasets/`, then run model training scripts, for example:

```bash
docker compose exec fastapi python ml_models/diabetes/train.py --data /app/datasets/diabetes.csv
```

## Required Environment Variables

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG`
- `DJANGO_ALLOWED_HOSTS`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `FASTAPI_URL`
- `DJANGO_URL`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_BACKEND_STORE_URI`
- `MLFLOW_ARTIFACT_ROOT`
- `QDRANT_HOST`
- `QDRANT_PORT`
- `QDRANT_COLLECTION`
- `COHERE_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`

## Architecture

The Django backend owns users, authentication, reports, health scores, timeline events, recommendations, knowledge-document records, and saved predictions. The FastAPI service owns model inference, report NLP, RAG indexing and retrieval, agent orchestration, and health-score calculations. Prometheus scrapes both API services, Grafana reads Prometheus, MLflow tracks model training and inference, and Nginx routes external traffic.

## Safety

MediMind AI provides educational support only. It does not diagnose, prescribe medication, or replace qualified clinical care. The emergency agent always runs before intent routing and cannot be bypassed by user instructions.