from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["health-score"])


class HealthScoreRequest(BaseModel):
    bmi: float | None = None
    glucose: float | None = None
    ldl: float | None = None
    cholesterol: float | None = None
    systolic: int | None = None
    diastolic: int | None = None
    smoking: bool | None = None
    exercise_frequency: str | None = None
    alcohol_use: str | None = None


@router.post("/calculate-health-score")
async def calculate_health_score(payload: HealthScoreRequest):
    try:
        base = 100
        strengths = []
        needs_improvement = []

        if payload.bmi is not None:
            if payload.bmi > 35:
                base -= 25
                needs_improvement.append("BMI is above 35.")
            elif 30 <= payload.bmi <= 35:
                base -= 15
                needs_improvement.append("BMI is in the obesity range.")
            elif 18.5 <= payload.bmi <= 24.9:
                base += 5
                strengths.append("BMI is in a healthy range.")

        if payload.glucose is not None:
            if payload.glucose > 126:
                base -= 20
                needs_improvement.append("Glucose is above the diabetes screening threshold.")
            elif 100 <= payload.glucose <= 125:
                base -= 10
                needs_improvement.append("Glucose is in the prediabetes range.")
            else:
                strengths.append("Glucose is in a healthier range.")

        ldl = payload.ldl if payload.ldl is not None else payload.cholesterol
        if ldl is not None and ldl > 160:
            base -= 15
            needs_improvement.append("LDL or cholesterol is elevated.")

        if payload.systolic is not None and payload.systolic > 140:
            base -= 15
            needs_improvement.append("Systolic blood pressure is elevated.")

        if payload.smoking:
            base -= 20
            needs_improvement.append("Smoking significantly increases cardiovascular risk.")

        if payload.exercise_frequency == "none":
            base -= 10
            needs_improvement.append("No exercise was reported.")
        elif payload.exercise_frequency == "daily":
            base += 5
            strengths.append("Daily exercise supports cardiometabolic health.")

        if payload.alcohol_use == "heavy":
            base -= 10
            needs_improvement.append("Heavy alcohol use can increase health risks.")

        score = max(0, min(100, round(base)))
        if score >= 85:
            category = "Excellent"
        elif score >= 70:
            category = "Good"
        elif score >= 50:
            category = "Fair"
        else:
            category = "Poor"

        return {
            "score": score,
            "strengths": strengths,
            "needs_improvement": needs_improvement,
            "category": category,
            "lifestyle_score": 10 if payload.exercise_frequency == "daily" else 0,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
