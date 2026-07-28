"""FastAPI router for Counterfactual Health Simulator."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any

from services.counterfactual import simulate_counterfactual
import metrics as m

router = APIRouter(prefix="/counterfactual", tags=["counterfactual"])


class ModificationInput(BaseModel):
    factor: str
    change: str = ""
    current: float | None = None
    target: float | None = None


class SimulateRequest(BaseModel):
    user_id: int
    original_risk: dict[str, float] = Field(
        ..., description='{"diabetes": 45.2, "heart": 30.1, ...}'
    )
    modifications: list[ModificationInput] = Field(
        ..., min_length=1, description="Changes to simulate"
    )
    user_profile: dict[str, Any] | None = None


@router.post("/simulate")
async def simulate(req: SimulateRequest) -> dict[str, Any]:
    """Simulate health outcomes under modified conditions."""
    mods = [m.model_dump() for m in req.modifications]
    result = simulate_counterfactual(
        original_risk=req.original_risk,
        modifications=mods,
        user_profile=req.user_profile,
    )
    affected = sum(1 for v in result.risk_deltas.values() if abs(v) > 0.1)
    m.counterfactual_simulations.labels(diseases_affected=str(affected)).inc()
    output = result.to_dict()
    output["user_id"] = req.user_id
    return output


@router.get("/available-factors")
async def available_factors() -> dict[str, Any]:
    """List the factors that can be modified in simulations."""
    from services.counterfactual import IMPACT_FACTORS

    factors: dict[str, Any] = {}
    for disease, dfactors in IMPACT_FACTORS.items():
        for factor, rules in dfactors.items():
            if factor not in factors:
                factors[factor] = {"diseases_affected": [], "changes": []}
            factors[factor]["diseases_affected"].append(disease)
            factors[factor]["changes"].extend(rules.keys())

    return {"factors": factors}
