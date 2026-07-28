# MediMind AI

**MediMind AI** is a production-oriented multi-agent healthcare platform combining a Django backend, a FastAPI AI service, and multiple ML/AI models for clinical decision support — including chest X-ray pneumonia detection with **Grad-CAM heatmap visualization**, disease risk prediction, an AI chat assistant, and a health score tracker.

---

## Features

| Feature | Description |
|---|---|
| **Chest X-ray Pneumonia Screening** | Upload a chest X-ray; CNN (VGG16) predicts NORMAL / PNEUMONIA with Grad-CAM heatmap overlay showing the regions that influenced the model's decision. |
| **Disease Risk Prediction** | Models for diabetes, heart disease, chronic kidney disease, and stroke — each with SHAP-based factor explanations and LLM-generated interpretations. |
| **AI Health Assistant** | Multi-agent chat (triage, symptom checker, health-education) powered by Groq / OpenRouter / Mistral. |
| **Health Score** | Calculated from user vitals (BMI, blood pressure, glucose, cholesterol) with lifestyle recommendations. |
| **Medical Reports** | Upload, compare, and view explainability for medical reports. |
| **Predictions Dashboard** | View all your risk assessments and X-ray analyses in one place. |
| **Timeline** | Chronological log of health events, scores, and assessments. |
| **Patient State Intelligence** | Aggregates all data sources (observations, predictions, meds, allergies) into a unified patient snapshot with priority triage and missing-data detection. |
| **Longitudinal Lab Intelligence** | Tracks lab values over time, detects trends (improving/worsening), flags sudden changes and persistent abnormalities. |
| **Chest X-ray Trust Gate** | Evaluates image quality, prediction confidence, and dataset similarity before surfacing results — automatically abstains when trust is insufficient. |
| **Counterfactual Health Simulator** | "What if" scenarios: model risk changes from lifestyle modifications (quit smoking, reduce glucose, exercise more). |
| **Medication Safety Passport** | Drug–drug interaction checks, allergy cross-reactivity detection, duplicate therapy alerts. |
| **Clinical Contradiction Detection** | Finds value discrepancies across lab sources and mismatches between prediction inputs and latest observations. |
| **Evidence-Linked Medical Assistant** | Every AI response carries numbered citations back to the observations, predictions, or reports that informed it. |
| **Clinician Review & Feedback** | Clinicians accept/modify/reject AI recommendations; model feedback (correct/incorrect/hallucination) feeds back for quality monitoring. |
| **FHIR R4 Patient Export** | Exports patient records as a FHIR R4 Bundle (Patient, Observation, RiskAssessment, AllergyIntolerance) for interoperability. |

---

## Quick Start (Local, no Docker)

The recommended way to run MediMind locally uses SQLite, the local `.venv`, and a single launcher script.

### Prerequisites

- Python 3.11+ (the project includes a `.venv` at the repository root)
- A `.env` file with LLM API keys (see [Environment Variables](#environment-variables))

### Setup

```bash
# 1. Activate the virtual environment
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate            # Windows

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run the application (migrates DB, seeds a preview account, starts both servers)
python run_local.py
```

The launcher starts:

| Service | URL | Purpose |
|---|---|---|
| Django | <http://127.0.0.1:8000> | Web UI, auth, reports, health score |
| FastAPI | <http://127.0.0.1:8001> | ML inference, AI agent, X-ray analysis |

**Login credentials:** `hammad` / `MediMind@12345`

---

## Docker Deployment

For a full production-like stack with PostgreSQL, Qdrant, MLflow, Prometheus, and Grafana:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Django API | <http://localhost:18000> |
| FastAPI AI | <http://localhost:18001> |
| Nginx gateway | <http://localhost:18080> |
| MLflow | <http://localhost:15000> |
| Prometheus | <http://localhost:19090> |
| Grafana | <http://localhost:13000> |
| Qdrant | <http://localhost:16333> |

---

## Chest X-ray Analysis with Grad-CAM

The CNN pneumonia screening module uses a **VGG16** backbone (fine-tuned on 2082 chest X-rays) wrapped in a Keras `Sequential` model with additional dense layers.

### How to use

1. Navigate to **Predictions → Chest X-ray Pneumonia**
2. Upload a JPEG / PNG / WEBP chest X-ray image (max 20 MB)
3. Click **Run Pneumonia Detection**
4. The result card shows:

   - **Prediction** — NORMAL or PNEUMONIA with confidence percentage
   - **Prediction Probabilities** — bar chart for both classes
   - **Grad-CAM Heatmap** — toggle between **Overlay** / **Heatmap** / **Original** views
   - **Clinical Interpretation** — explanatory text about the highlighted regions
   - **Clinical Recommendations** — next-step guidance
   - **Assessment Summary** — natural-language report

### Grad-CAM technical details

- **Target layer:** `block5_conv3` (VGG16's last Conv2D, output shape 14×14×512)
- **Algorithm:** Gradients of the predicted-class score w.r.t. conv feature maps are global-average-pooled, used to weight each feature map, summed, ReLU'd, and normalized
- **Overlay:** Jet colormap composited at 40% transparency on the original image at its native resolution
- **Fallback:** When the trained CNN model file is unavailable, a dataset-nearest-neighbor fallback is used (Grad-CAM shows `unavailable`)

---

## Disease Risk Prediction Models

Four ML models (trained with scikit-learn / XGBoost) provide risk assessments:

| Model | Features | Interpretability |
|---|---|---|
| **Diabetes** | glucose, BMI, age, blood pressure, insulin, etc. | SHAP feature importance + LLM explanation |
| **Heart Disease** | chest pain type, resting BP, cholesterol, max heart rate, etc. | SHAP feature importance + LLM explanation |
| **Chronic Kidney Disease** | serum creatinine, hemoglobin, albumin, blood urea, etc. | SHAP feature importance + LLM explanation |
| **Stroke** | age, hypertension, glucose level, BMI, smoking status, etc. | SHAP feature importance + LLM explanation |

### Training local models

```bash
.venv/Scripts/python.exe ai_service/ml_models/train_all_local.py
```

This generates synthetic training data (where no real dataset exists) and trains all four models, saving artifacts (`model.joblib`, `scaler.joblib`, `shap_explainer.joblib`, `feature_columns.json`) per disease.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (18080)                         │
│              Reverse proxy, SSL, static files             │
└────┬────────────────────────────────────┬────────────────┘
     │                                    │
┌────▼──────────────┐          ┌──────────▼──────────────┐
│  Django Backend   │          │   FastAPI AI Service    │
│  :18000 / :8000   │          │    :18001 / :8001       │
│                   │          │                          │
│ • Users / Auth    │◄────────►│ • CNN pneumonia (VGG16)  │
│ • Health Score    │  HTTP    │ • Grad-CAM heatmaps      │
│ • Reports         │  sync    │ • Trust Gate (OOD +      │
│ • Timeline        │          │   confidence + quality)  │
│ • Recommendations │          │ • Disease risk models    │
│ • Knowledge docs  │          │   (diabetes, heart,      │
│ • Saved predictions│         │    kidney, stroke)       │
│                   │          │ • Patient State Engine   │
│ CLINICAL APPS     │          │ • Lab Trend Analysis     │
│ • Clinical Intel  │          │ • Medication Safety      │
│ • Medications     │          │ • Counterfactual Sim     │
│ • Lab Trends      │          │ • Conflict Detection     │
│ • Reviews         │          │ • Evidence Tracker       │
│ • FHIR Export     │          │ • FHIR Export            │
│                   │          │ • Multi-agent AI chat    │
└────┬───────────────┘          • RAG (Qdrant vector DB)  │
     │                         └──────────┬──────────────┘
     │                                    │
┌────▼──────────────┐          ┌──────────▼──────────────┐
│   PostgreSQL      │          │    MLflow (15000)        │
│   (Docker only)   │          │  Model registry &        │
│                   │          │  experiment tracking     │
└───────────────────┘          └─────────────────────────┘
```

### Local mode (run_local.py)

- **Database:** SQLite (no PostgreSQL needed)
- **LLM:** Uses Groq / OpenRouter / Mistral API keys from `.env`
- **MLflow:** Disabled by default (`DISABLE_MLFLOW=true`)
- **Qdrant:** Disabled by default (`DISABLE_QDRANT=true`)
- **No Docker required**

---

## API Reference

### FastAPI AI Service (`http://127.0.0.1:8001`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, model statuses |
| `POST` | `/cnn/predict/{model_id}` | Upload X-ray → prediction + Grad-CAM heatmap |
| `GET` | `/cnn/models` | List loaded CNN models |
| `POST` | `/predict/{disease}` | Disease risk prediction (diabetes, heart, kidney, stroke) |
| `POST` | `/chat` | Multi-agent AI chat |
| `POST` | `/reports/extract` | Extract structured data from a medical report |
| `POST` | `/reports/compare` | Compare two medical reports |
| `GET` | `/rag/search?query=...` | Knowledge-base search (requires Qdrant) |

### Clinical Intelligence Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/lab/analyze-trend` | Analyze longitudinal trends for a single test |
| `POST` | `/lab/analyze-all` | Analyze trends across all tests for a patient |
| `POST` | `/lab/classify-abnormality` | Classify an observation against its reference range |
| `POST` | `/patient-state/build` | Build a comprehensive patient state snapshot |
| `POST` | `/patient-state/emergency-check` | Check text for emergency keywords |
| `POST` | `/medication/safety-check` | Drug interactions, allergy conflicts, duplicate therapy |
| `POST` | `/medication/extract-from-text` | Extract medications from free text |
| `POST` | `/counterfactual/simulate` | Simulate health outcomes under modified conditions |
| `GET` | `/counterfactual/available-factors` | List modifiable factors and their diseases |
| `POST` | `/conflicts/detect` | Detect value discrepancies and cross-source conflicts |
| `POST` | `/review/build-evidence` | Build evidence-linked response with provenance |
| `POST` | `/review/record-feedback` | Submit model feedback (correct/incorrect/hallucination) |
| `POST` | `/review/record-decision` | Record clinician review decision |
| `POST` | `/fhir/export` | Generate a FHIR R4 Bundle from patient data |
| `GET` | `/fhir/capabilities` | FHIR capability statement |

### Example: Chest X-ray Prediction

```bash
curl -X POST http://127.0.0.1:8001/cnn/predict/pneumonia_xray \
  -F "file=@chest_xray.jpeg"
```

Response includes:

```json
{
  "predicted_class": "NORMAL",
  "confidence_percentage": 99.79,
  "probabilities": { "NORMAL": 0.9979, "PNEUMONIA": 0.0021 },
  "gradcam": {
    "status": "generated",
    "heatmap_base64": "data:image/png;base64,...",
    "overlay_base64": "data:image/png;base64,...",
    "layer_used": "block5_conv3",
    "explanation": "The highlighted regions (yellow/red)..."
  },
  "clinical_recommendations": [...],
  "formatted_report": "..."
}
```

---

## Project Structure

```
medimind-ai/
├── ai_service/                 # FastAPI AI service
│   ├── app.py                  # FastAPI entrypoint
│   ├── metrics.py              # Prometheus counters/histograms
│   ├── cnn/                    # Chest X-ray pipeline
│   │   ├── config.py           # Model configs & paths
│   │   ├── gradcam.py          # Grad-CAM heatmap generator
│   │   ├── trust_gate.py       # Trust scoring (quality + confidence + OOD)
│   │   ├── registry.py         # Model registry + predict + trust logic
│   │   ├── preprocessing.py    # Image validation + OOD detection
│   │   ├── fallback.py         # kNN fallback (BLOCKED for medical)
│   │   └── diagnose.py         # Diagnostic utilities
│   ├── routers/                # API route handlers
│   │   ├── cnn_inference.py    # /cnn/predict endpoint
│   │   ├── prediction.py       # Disease risk prediction
│   │   ├── lab_intelligence.py # /lab/analyze-trend, /lab/analyze-all
│   │   ├── patient_state.py    # /patient-state/build, emergency-check
│   │   ├── medication_safety.py# /medication/safety-check
│   │   ├── counterfactual.py   # /counterfactual/simulate
│   │   ├── conflict_detection.py # /conflicts/detect
│   │   ├── clinician_review.py # /review/build-evidence, record-feedback
│   │   ├── fhir_export.py      # /fhir/export, /fhir/capabilities
│   │   └── ...                 # (chat, reports, rag, comparison, etc.)
│   ├── services/               # Business logic layer
│   │   ├── lab_intelligence.py # Trend analysis, abnormality classification
│   │   ├── patient_state.py    # Patient state aggregation, emergency triage
│   │   ├── medication_safety.py# Drug interactions, allergy cross-reactivity
│   │   ├── counterfactual.py   # What-if health scenario modeling
│   │   ├── conflict_detection.py # Cross-source contradiction detection
│   │   ├── evidence_tracker.py # Evidence-linked response builder
│   │   ├── fhir_export.py      # FHIR R4 resource mapping + export
│   │   └── django_client.py    # Django persistence bridge
│   ├── ml_models/              # Disease models (diabetes, heart, kidney, stroke)
│   ├── agents/                 # Multi-agent system (triage override)
│   ├── llm/                    # LLM provider abstraction
│   └── rag/                    # RAG indexing + search
├── backend/django/             # Django web application
│   ├── medimind/               # Django project config + URL routing
│   ├── clinical/               # Clinical observations, conflicts, snapshots
│   ├── medication/             # Medications, safety alerts, allergies
│   ├── reviews/                # Review decisions, model feedback, audit events
│   ├── api/                    # DRF API views for clinical, medication, reviews
│   ├── templates/              # Jinja2 templates
│   │   ├── dashboard/          # Health dashboard
│   │   ├── predictions/        # Disease forms + X-ray (with trust gate)
│   │   ├── clinical/           # Clinical Intel, Lab Trends, Medications,
│   │   │                       # Simulator, FHIR Export, Reviews
│   │   ├── auth/               # Login / register
│   │   ├── health_score/       # Health score UI
│   │   ├── reports/            # Medical reports
│   │   ├── assistant/          # AI chat UI
│   │   ├── timeline/           # Event timeline
│   │   └── comparison/         # Report comparison
│   └── static/css/             # Design system (single-hue green)
├── tests/
│   └── test_clinical_services.py # 51 pytest tests for all services
├── data/                       # Datasets (X-rays + tabular)
├── ml/                         # CNN model artifacts
├── run_local.py                # Local dev launcher
└── requirements.txt            # All Python dependencies
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DJANGO_SECRET_KEY` | Yes | — | Django secret key |
| `DJANGO_DEBUG` | No | `True` | Debug mode |
| `GROQ_API_KEY` | Yes* | — | Primary LLM (llama-3.3-70b) |
| `OPENROUTER_API_KEY` | No | — | Fallback LLM |
| `MISTRAL_API_KEY` | No | — | Alternative LLM |
| `COHERE_API_KEY` | No | — | Embeddings (RAG) |
| `POSTGRES_DB` | Docker | `medimind_db` | PostgreSQL database |
| `POSTGRES_USER` | Docker | `medimind_user` | PostgreSQL user |
| `POSTGRES_PASSWORD` | Docker | — | PostgreSQL password |
| `QDRANT_HOST` | Docker | `qdrant` | Vector DB host |
| `MLFLOW_TRACKING_URI` | Docker | `http://mlflow:5000` | MLflow server |

*\* At least one LLM API key (Groq, OpenRouter, or Mistral) is required for the AI chat assistant and LLM-generated explanations. The CNN prediction and disease risk models work without any LLM key.*

---

## Observability

Prometheus metrics are exposed at `/metrics` on the FastAPI service. Clinical intelligence counters track:

| Metric | Description |
|---|---|
| `medimind_lab_trend_analyses_total` | Lab trend analyses performed |
| `medimind_patient_state_builds_total` | Patient state snapshots built (by priority) |
| `medimind_emergency_checks_total` | Emergency triage checks (by result) |
| `medimind_medication_safety_checks_total` | Medication safety checks (by alert presence) |
| `medimind_medication_alerts_total` | Individual alerts (by type and severity) |
| `medimind_counterfactual_simulations_total` | Counterfactual simulations (by diseases affected) |
| `medimind_conflict_detections_total` | Conflict detection runs |
| `medimind_trust_gate_assessments_total` | CNN trust gate assessments (by status) |
| `medimind_review_decisions_total` | Clinician review decisions |
| `medimind_model_feedback_total` | Model feedback events (by type and model) |
| `medimind_fhir_exports_total` | FHIR bundle exports |

In Docker mode, Prometheus scrapes the FastAPI service and Grafana provides pre-configured dashboards at `http://localhost:13000`.

---

## Safety

**MediMind AI provides educational decision support only.** It does not diagnose, prescribe medication, or replace qualified clinical care. All AI-generated assessments include disclaimers reminding users that results must be reviewed by a qualified clinician.

### Clinical Safety Requirements

| Requirement | Implementation |
|---|---|
| **No nearest-neighbor fallback for medical predictions** | `_load_dataset_fallback()` in `registry.py` blocks kNN entirely — model is marked unavailable with a clear error message |
| **No silent model replacement** | If the trained CNN model file is missing, the system raises `CNNModelUnavailable` with explicit guidance instead of substituting a proxy |
| **Deterministic vs. LLM separation** | Lab trend analysis, abnormality classification, counterfactual simulation, medication safety checks, and conflict detection are all deterministic (no LLM). LLM is used only for narrative interpretation. |
| **Emergency triage override** | `check_emergency_keywords()` runs before any intent routing in the multi-agent system and cannot be bypassed |
| **Synthetic data labeling** | All model outputs and patient state snapshots include `is_synthetic_data` flags |
| **Disclaimers everywhere** | Every API response, page view, and export includes appropriate disclaimers about educational use |
| **Trust Gate** | CNN predictions are gated through image quality, confidence, and dataset similarity checks — results are abstained when trust is insufficient |
| **OOD detection** | Non-chest-X-ray images are rejected before inference via grayscale ratio, aspect ratio, edge density, and histogram spread checks |
| **Clinician review loop** | All AI outputs can be accepted/modified/rejected by clinicians; feedback is tracked for quality monitoring |

### What MediMind AI does NOT do

- Make autonomous diagnoses or treatment decisions
- Replace clinical judgment or professional medical advice
- Guarantee accuracy of any prediction or assessment
- Operate without clinical oversight (review workflow required for any clinical use)
