"""Prometheus metrics for MediMind clinical intelligence features."""

from prometheus_client import Counter, Histogram, Gauge

# ── Clinical Intelligence Metrics ────────────────────────────────────────────

# Lab Intelligence
lab_trend_analyses = Counter(
    "medimind_lab_trend_analyses_total",
    "Total longitudinal lab trend analyses performed",
    ["status"],
)
lab_trend_duration = Histogram(
    "medimind_lab_trend_duration_seconds",
    "Time to compute lab trend analysis",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Patient State
patient_state_builds = Counter(
    "medimind_patient_state_builds_total",
    "Total patient state snapshots built",
    ["priority_level"],
)
emergency_checks = Counter(
    "medimind_emergency_checks_total",
    "Emergency keyword triage checks",
    ["is_emergency"],
)

# Medication Safety
medication_safety_checks = Counter(
    "medimind_medication_safety_checks_total",
    "Total medication safety checks performed",
    ["has_alerts"],
)
medication_alerts_generated = Counter(
    "medimind_medication_alerts_total",
    "Individual medication safety alerts generated",
    ["alert_type", "severity"],
)

# Counterfactual Simulator
counterfactual_simulations = Counter(
    "medimind_counterfactual_simulations_total",
    "Total counterfactual health simulations",
    ["diseases_affected"],
)
counterfactual_duration = Histogram(
    "medimind_counterfactual_duration_seconds",
    "Time to run counterfactual simulation",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)

# Conflict Detection
conflict_detections = Counter(
    "medimind_conflict_detections_total",
    "Total conflict detection runs",
    ["has_conflicts"],
)
conflicts_found = Counter(
    "medimind_conflicts_found_total",
    "Individual conflicts detected",
    ["conflict_type", "severity"],
)

# Trust Gate
trust_gate_assessments = Counter(
    "medimind_trust_gate_assessments_total",
    "CNN prediction trust gate assessments",
    ["trust_status"],
)

# Clinician Review
review_decisions = Counter(
    "medimind_review_decisions_total",
    "Clinician review decisions",
    ["decision_type"],
)
model_feedback_events = Counter(
    "medimind_model_feedback_total",
    "Model feedback events submitted",
    ["feedback_type", "model_name"],
)

# FHIR Export
fhir_exports = Counter(
    "medimind_fhir_exports_total",
    "FHIR bundle exports",
    ["resource_count_bucket"],
)

# System health
active_patients_tracked = Gauge(
    "medimind_active_patients_tracked",
    "Number of patients with current state snapshots",
)
models_loaded = Gauge(
    "medimind_ai_models_loaded",
    "Number of ML models loaded in registry",
)
cnn_models_loaded = Gauge(
    "medimind_cnn_models_loaded",
    "Number of CNN models loaded",
)
