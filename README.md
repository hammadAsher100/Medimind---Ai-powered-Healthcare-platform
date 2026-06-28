<div align="center">

<br>

```
███╗   ███╗███████╗██████╗ ██╗███╗   ███╗██╗███╗   ██╗██████╗      █████╗ ██╗
████╗ ████║██╔════╝██╔══██╗██║████╗ ████║██║████╗  ██║██╔══██╗    ██╔══██╗██║
██╔████╔██║█████╗  ██║  ██║██║██╔████╔██║██║██╔██╗ ██║██║  ██║    ███████║██║
██║╚██╔╝██║██╔══╝  ██║  ██║██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║    ██╔══██║██║
██║ ╚═╝ ██║███████╗██████╔╝██║██║ ╚═╝ ██║██║██║ ╚████║██████╔╝    ██║  ██║██║
╚═╝     ╚═╝╚══════╝╚═════╝ ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝
```

### An Intelligent Multi-Agent Healthcare Platform

*Disease Risk Assessment · Medical Report Analysis · Personalized Health Assistance*

<br>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.0-092E20?style=for-the-badge&logo=django&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-embed--english--v3.0-39594E?style=for-the-badge&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=for-the-badge&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge&logoColor=white)

<br>

> ⚠️ **Medical Disclaimer** — MediMind AI is an educational tool only. It does not diagnose, prescribe, or replace professional medical advice. Always consult a qualified healthcare provider.

<br>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Platform Modules](#platform-modules)
- [System Architecture](#system-architecture)
- [Multi-Agent System](#multi-agent-system)
- [ML Disease Models](#ml-disease-models)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [AI Explainability](#ai-explainability)
- [Monitoring](#monitoring)

---

## Overview

MediMind AI is a production-grade, multi-agent healthcare platform built for a hackathon. It combines classical machine learning (disease risk prediction with SHAP explainability), retrieval-augmented generation (WHO guidelines + medical literature), and a coordinated agent system that routes every user request to the right specialist.

The platform runs as two decoupled services — a **Django backend** for auth, data, and CRUD, and a **FastAPI AI service** for all inference, agents, and RAG — orchestrated with Docker Compose and fronted by Nginx.

---

## Platform Modules

| # | Module | Description |
|---|--------|-------------|
| 01 | **Authentication** | JWT auth, user registration, medical profiles, allergy tracking, family history |
| 02 | **Health Dashboard** | BMI, weight history, health score, risk factors, recent reports, AI recommendations |
| 03 | **Disease Risk Prediction** | ML models for diabetes, heart disease, kidney disease, stroke — with SHAP explanations |
| 04 | **Medical Report Analyzer** | Upload blood/lab PDFs; AI extracts and explains cholesterol, HDL, LDL, glucose in plain language |
| 05 | **Knowledge Base (RAG)** | WHO guidelines, medical books, research papers — chunked, embedded, stored in Qdrant |
| 06 | **AI Medical Assistant** | Coordinator routes requests to 7 specialist agents |
| 07 | **Conversation Memory** | Persistent chat history and prediction context stored in PostgreSQL |
| 08 | **Health Timeline** | Every report, prediction, and recommendation becomes a reviewable timeline event |
| 09 | **AI Health Score** | Composite 0–100 score from BMI, sugar, cholesterol, blood pressure, and lifestyle |
| 10 | **Progress Comparison** | Upload a new report and AI compares it to previous results with trend analysis |
| 11 | **MLOps Dashboard** | Model versions, accuracy, precision, recall, experiments via MLflow — admin only |
| 12 | **Monitoring** | Response time, API usage, prediction counts, error rates via Prometheus + Grafana |
| 13 | **Deployment** | Full Docker Compose stack: Nginx → Django → FastAPI → PostgreSQL → MLflow → Prometheus → Grafana |

---

## System Architecture

```
                          ┌─────────────────────────────────────┐
                          │              NGINX : 80              │
                          │   /api/ → Django  /ai/ → FastAPI    │
                          │   /mlflow/        /grafana/          │
                          └────────────┬────────────┬───────────┘
                                       │            │
              ┌────────────────────────▼──┐    ┌───▼───────────────────────────┐
              │     DJANGO : 8000         │    │       FASTAPI : 8001           │
              │                           │    │                                │
              │  ┌─────────────────────┐  │    │  ┌──────────────────────────┐ │
              │  │  authentication/    │  │    │  │    Coordinator Agent     │ │
              │  │  users/             │  │◄───┤  │    ┌──────────────────┐  │ │
              │  │  reports/           │  │    │  │    │ Emergency Check  │  │ │
              │  │  health_score/      │  │    │  │    └────────┬─────────┘  │ │
              │  │  timeline/          │  │    │  │             │             │ │
              │  │  dashboard/         │  │    │  │   ┌─────────▼──────────┐ │ │
              │  └─────────────────────┘  │    │  │   │  Intent Classifier │ │ │
              └────────────┬──────────────┘    │  │   └─────────┬──────────┘ │ │
                           │                   │  │             │             │ │
              ┌────────────▼──────────────┐    │  │  ┌──────────▼──────────┐ │ │
              │      POSTGRESQL : 5432    │    │  │  │   Specialist Agents │ │ │
              │                           │    │  │  │  Diagnosis · Report  │ │ │
              │  Users · Profiles         │◄───┤  │  │  Nutrition · Meds   │ │ │
              │  Reports · Predictions    │    │  │  │  Lifestyle · Memory  │ │ │
              │  Timeline · Memory        │    │  │  └──────────┬──────────┘ │ │
              └───────────────────────────┘    │  └────────────┼─────────────┘ │
                                               └───────────────┼───────────────┘
              ┌────────────────────────────────────────────────┼───────────────┐
              │                  INTELLIGENCE LAYER            │               │
              │                                                │               │
              │   ┌──────────────┐  ┌──────────────┐  ┌───────▼────────────┐  │
              │   │ GROQ LLaMA   │  │   COHERE     │  │      QDRANT        │  │
              │   │ 3.3-70B      │  │ embed-v3.0   │  │  medimind_knowledge│  │
              │   │ (primary)    │  │ (1024-dim)   │  │  Cosine Similarity │  │
              │   └──────────────┘  └──────────────┘  └────────────────────┘  │
              │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
              │   │ OPENROUTER   │  │    MLFLOW    │  │    ML MODELS       │  │
              │   │ Mistral-7B   │  │ Experiments  │  │  diabetes · heart  │  │
              │   │ (fallback)   │  │ Model Registry│  │  kidney · stroke   │  │
              │   └──────────────┘  └──────────────┘  └────────────────────┘  │
              └───────────────────────────────────────────────────────────────┘
```

### Request flow

```
Browser → Nginx → Django (auth) → FastAPI → Coordinator Agent → Specialist Agent → Groq LLM → Response
```

---

## Multi-Agent System

The AI assistant is **not a chatbot** — it is an orchestrator. Every request passes through the Coordinator, which routes to one or more specialist agents running in parallel.

```
User Message
     │
     ▼
┌────────────────────────────────────┐
│         COORDINATOR AGENT          │
│                                    │
│  1. Emergency keyword scan  ──────►│──► EMERGENCY AGENT (immediate)
│  2. Intent classification          │
│  3. Route to specialist(s)         │
│  4. Merge + format response        │
└────┬───────────────────────────────┘
     │
     ├──► DIAGNOSIS AGENT    — symptoms + ML predictions + RAG retrieval
     ├──► REPORT AGENT       — explains stored lab report analysis
     ├──► MEDICATION AGENT   — explains medicines, never prescribes
     ├──► NUTRITION AGENT    — personalized meal plans from medical profile
     ├──► LIFESTYLE AGENT    — exercise, sleep, hydration, weight goals
     └──► MEMORY AGENT       — retrieves last 10 conversation turns
```

**Emergency detection** runs before any other logic. If a user mentions chest pain, difficulty breathing, loss of consciousness, or stroke symptoms, the Emergency Agent responds immediately with crisis guidance and logs a `emergency_detected` timeline event. This cannot be bypassed.

**Medication Agent constraint** — hardcoded in the system prompt: never suggests dosages, never prescribes. Always ends with: *"Follow your doctor's prescription."*

---

## ML Disease Models

Each model follows the same rigorous pipeline:

```
Raw Dataset
    │
    ▼
EDA (nulls, class distribution, correlation heatmap)
    │
    ▼
Preprocessing (impute → encode → StandardScaler → SMOTE)
    │
    ▼
Feature Engineering (disease-specific derived features)
    │
    ▼
Model Selection (LR · RandomForest · GradientBoosting · XGBoost)
    │  StratifiedKFold, 5 folds
    ▼
Best Model → SHAP Explainability → MLflow Logging
    │
    ▼
Artifacts saved: model.joblib · scaler.joblib · feature_columns.json · shap_explainer.joblib
```

| Disease | Dataset | Key Features Engineered | Best Model (typical) |
|---------|---------|------------------------|----------------------|
| **Diabetes** | Pima Indians (UCI) | BMI category, age group, glucose category | XGBoost |
| **Heart Disease** | Cleveland (UCI) | Cholesterol ratio, thalach/age interaction | GradientBoosting |
| **Kidney Disease** | CKD Dataset (UCI) | Creatinine/BUN ratio, anemia flag | RandomForest |
| **Stroke** | Kaggle Stroke | Hypertension + heart disease combined flag | XGBoost |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Django 5 + Django REST Framework + SimpleJWT |
| AI Microservice | FastAPI + Uvicorn |
| Database | PostgreSQL 15 |
| Primary LLM | Groq — `llama-3.3-70b-versatile` |
| Fallback LLM | OpenRouter — `mistralai/mistral-7b-instruct` |
| Embeddings | Cohere — `embed-english-v3.0` (1024-dim) |
| Vector Database | Qdrant — Cosine similarity |
| ML Training | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP (TreeExplainer / LinearExplainer) |
| Experiment Tracking | MLflow 2.13 |
| PDF Extraction | pdfplumber |
| Retry Logic | tenacity — 3 attempts, exponential backoff |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + docker-compose (8 services) |
| Reverse Proxy | Nginx |

---

## Project Structure

```
medimind-ai/
│
├── backend/
│   └── django/
│       ├── medimind/               Django project settings (base / dev / prod)
│       ├── authentication/         JWT auth, register, login, logout
│       ├── users/                  MedicalProfile, Allergy, FamilyHistory
│       ├── reports/                PDF upload, analysis storage
│       ├── health_score/           Score calculation + history
│       ├── timeline/               Event log for all user actions
│       ├── dashboard/              Aggregated health overview
│       └── recommendations/        AI recommendation storage
│
├── ai_service/
│   ├── app.py                      FastAPI entrypoint + startup events
│   ├── routers/
│   │   ├── prediction.py           POST /predict/{disease}
│   │   ├── report_analyzer.py      POST /analyze-report
│   │   ├── health_score.py         POST /calculate-health-score
│   │   ├── comparison.py           POST /compare-reports
│   │   └── knowledge_base.py       POST /index-document
│   │
│   ├── agents/
│   │   ├── coordinator.py          Master router + emergency detection
│   │   ├── diagnosis.py            Symptom analysis + RAG + ML context
│   │   ├── report.py               Lab report explanation
│   │   ├── medication.py           Medicine info (never prescribes)
│   │   ├── nutrition.py            Personalized meal plans
│   │   ├── lifestyle.py            Exercise, sleep, hydration
│   │   ├── emergency.py            Crisis detection + immediate response
│   │   └── memory.py               Conversation history management
│   │
│   ├── rag/
│   │   ├── embeddings/             Cohere embed-english-v3.0 client
│   │   ├── vector_store/           Qdrant upsert + search
│   │   ├── chunking/               Sentence-boundary text splitter
│   │   ├── loaders/                PDF loader + metadata extraction
│   │   └── retrieval/              Query → embed → Qdrant → top-k chunks
│   │
│   ├── ml_models/
│   │   ├── diabetes/               model.joblib · scaler.joblib · shap_explainer.joblib
│   │   ├── heart/
│   │   ├── kidney/
│   │   └── stroke/
│   │
│   ├── llm/
│   │   ├── provider.py             Routes to Groq or OpenRouter
│   │   ├── groq_client.py          Groq SDK wrapper + retry
│   │   └── openrouter_client.py    OpenRouter requests wrapper + retry
│   │
│   └── mlflow_utils/               Logging helpers for training + inference
│
├── docker/
│   ├── django.Dockerfile
│   ├── fastapi.Dockerfile
│   └── nginx.Dockerfile
│
├── nginx/
│   └── nginx.conf                  Routes /api/ /ai/ /mlflow/ /grafana/
│
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/datasources.yml
│
├── notebooks/
│   ├── 01_diabetes_eda_training.ipynb
│   ├── 02_heart_eda_training.ipynb
│   ├── 03_kidney_eda_training.ipynb
│   └── 04_stroke_eda_training.ipynb
│
├── datasets/                       Download instructions per dataset
├── docs/architecture.md
├── docker-compose.yml              Orchestrates all 8 containers
├── requirements.txt                All Python dependencies (pinned)
└── .env                            ← never commit this
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- API keys for Cohere, Groq, and OpenRouter

### 1 — Clone the repo

```bash
git clone https://github.com/yourname/medimind-ai.git
cd medimind-ai
```

### 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
COHERE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
DB_PASSWORD=choose_a_strong_password
SECRET_KEY=generate_with_django_get_random_secret_key
DJANGO_SERVICE_TOKEN=generate_with_openssl_rand_hex_32
```

### 3 — Train ML models (one-time)

Run locally before building Docker:

```bash
pip install -r requirements.txt

python ai_service/ml_models/diabetes/train.py
python ai_service/ml_models/heart/train.py
python ai_service/ml_models/kidney/train.py
python ai_service/ml_models/stroke/train.py
```

Or use the notebooks in `notebooks/` for full EDA + training.

### 4 — Start all services

```bash
docker-compose up --build
```

This spins up 8 containers: PostgreSQL, Qdrant, MLflow, Django, FastAPI, Prometheus, Grafana, Nginx.

### 5 — Initialize Django

```bash
docker-compose exec django python manage.py migrate
docker-compose exec django python manage.py createsuperuser
docker-compose exec django python manage.py collectstatic --noinput
```

### 6 — Access the platform

| Service | URL |
|---------|-----|
| Main App | http://localhost/ |
| FastAPI Docs | http://localhost/ai/docs |
| Django Admin | http://localhost/api/admin/ |
| MLflow UI | http://localhost/mlflow/ |
| Grafana | http://localhost/grafana/ |
| Prometheus | http://localhost:9090/ |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | ✅ | — | Django secret key |
| `DEBUG` | ✅ | `True` | Set `False` in production |
| `DB_PASSWORD` | ✅ | — | PostgreSQL password |
| `COHERE_API_KEY` | ✅ | — | Cohere embeddings API key |
| `GROQ_API_KEY` | ✅ | — | Groq LLM API key |
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter fallback LLM key |
| `DJANGO_SERVICE_TOKEN` | ✅ | — | Internal auth token (FastAPI → Django) |
| `DB_NAME` | ➖ | `medimind_db` | PostgreSQL database name |
| `DB_USER` | ➖ | `medimind_user` | PostgreSQL username |
| `DB_HOST` | ➖ | `postgres` | PostgreSQL host (Docker service name) |
| `QDRANT_HOST` | ➖ | `qdrant` | Qdrant host |
| `QDRANT_PORT` | ➖ | `6333` | Qdrant port |
| `QDRANT_COLLECTION` | ➖ | `medimind_knowledge` | Qdrant collection name |
| `MLFLOW_TRACKING_URI` | ➖ | `http://mlflow:5000` | MLflow tracking server |
| `FASTAPI_URL` | ➖ | `http://fastapi:8001` | FastAPI internal URL |
| `GRAFANA_ADMIN_PASSWORD` | ➖ | `admin` | Grafana admin password |

> **Never commit `.env` to git.** It is listed in `.gitignore` by default.

---

## API Endpoints

### Django REST API (`/api/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register/` | Register new user |
| `POST` | `/api/auth/login/` | Login, returns JWT pair |
| `POST` | `/api/auth/refresh/` | Refresh access token |
| `GET/PUT` | `/api/users/profile/` | Medical profile CRUD |
| `GET/POST` | `/api/users/allergies/` | Allergy management |
| `GET/POST` | `/api/users/family-history/` | Family history |
| `GET` | `/api/dashboard/` | Aggregated health overview |
| `POST` | `/api/reports/upload/` | Upload PDF report |
| `GET` | `/api/reports/` | List all reports |
| `GET/PUT` | `/api/health-score/` | Get or recalculate health score |
| `GET` | `/api/health-score/history/` | Last 12 scores for charting |
| `GET` | `/api/timeline/` | Full health timeline |
| `POST` | `/api/knowledge-base/upload/` | Admin: index medical document |

### FastAPI AI Service (`/ai/`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai/predict/diabetes` | Diabetes risk prediction + SHAP |
| `POST` | `/ai/predict/heart` | Heart disease risk + SHAP |
| `POST` | `/ai/predict/kidney` | Kidney disease risk + SHAP |
| `POST` | `/ai/predict/stroke` | Stroke risk + SHAP |
| `POST` | `/ai/analyze-report` | Extract + explain PDF lab report |
| `POST` | `/ai/calculate-health-score` | Compute composite health score |
| `POST` | `/ai/compare-reports` | Diff two reports with AI narrative |
| `POST` | `/ai/chat` | Multi-agent assistant |
| `POST` | `/ai/index-document` | Chunk + embed + store in Qdrant |
| `GET` | `/ai/metrics` | Prometheus metrics endpoint |
| `GET` | `/ai/docs` | Auto-generated Swagger UI |

---

## AI Explainability

Every disease prediction includes a full SHAP breakdown — not just a risk percentage, but *why*.

```
Prediction: High Diabetes Risk (87%)

Top contributing factors:

  Blood Glucose    ████████████████████████  +42.1%  ↑ increases risk
  BMI              ████████████████          +25.3%  ↑ increases risk
  Age              ████████████              +18.0%  ↑ increases risk
  Family History   ██████                   +10.2%  ↑ increases risk
  Blood Pressure   ███                       -4.4%  ↓ decreases risk

AI Recommendation:
Your elevated blood glucose is the primary driver of this prediction.
Reducing fasting glucose through dietary changes and regular monitoring
would have the highest impact on lowering your risk score.
```

The Explainability page renders a Chart.js horizontal bar chart with red bars for risk-increasing factors and green bars for protective factors, followed by a Groq-generated plain-English explanation and RAG-retrieved disease context.

---

## Monitoring

Prometheus scrapes metrics from FastAPI every 15 seconds. Grafana is pre-configured with Prometheus as the default datasource.

**Key metrics tracked:**

- `http_requests_total` — request count by endpoint and status
- `http_request_duration_seconds` — latency percentiles
- `prediction_requests_total` — ML inference count by disease
- `llm_call_duration_seconds` — Groq / OpenRouter latency
- `rag_retrieval_duration_seconds` — Qdrant query latency

Access Grafana at `http://localhost/grafana/` with credentials from `.env` (`admin` / `admin` by default).

---

## Hard Constraints

These are enforced in code, not just guidelines:

- **No LangChain** — all agent logic, RAG, and LLM calls are implemented directly
- **No OpenAI** — Groq, OpenRouter (Mistral), and Cohere only
- **No hardcoded secrets** — everything via `.env`
- **Emergency agent always runs first** — before intent classification, before any other routing
- **Medication agent never prescribes** — enforced as a hard system prompt constraint; no dosages, no prescriptions
- **Retry on all LLM calls** — 3 attempts with exponential backoff via `tenacity`

---

<div align="center">

Built with Django 5 · FastAPI · Groq · Cohere · Qdrant · MLflow · SHAP · Docker

**MediMind AI is not a substitute for professional medical advice.**

</div>