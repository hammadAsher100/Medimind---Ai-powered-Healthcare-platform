from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.coordinator import route_message

router = APIRouter(tags=["agents"])


class AgentChatRequest(BaseModel):
    user_id: int
    user_message: str
    medical_profile: dict = {}
    medical_history: list = []
    recent_predictions: list = []
    recent_reports: list = []


@router.post("/agents/chat")
async def chat(payload: AgentChatRequest):
    try:
        return await route_message(
            user_id=payload.user_id,
            user_message=payload.user_message,
            medical_profile=payload.medical_profile,
            medical_history=payload.medical_history,
            recent_predictions=payload.recent_predictions,
            recent_reports=payload.recent_reports,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
