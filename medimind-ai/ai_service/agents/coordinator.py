import asyncio
import json

from llm.provider import LLMProvider
from rag.retrieval.retriever import Retriever

from .base import AgentContext, AgentResponse
from .diagnosis import DiagnosisAgent
from .emergency import EmergencyAgent, is_emergency
from .lifestyle import LifestyleAgent
from .medication import MedicationAgent
from .memory import get_recent_memory, log_timeline_event, save_conversation_turn, summarize_if_needed
from .nutrition import NutritionAgent
from .report import ReportAgent

AGENTS = {
    "diagnosis": DiagnosisAgent,
    "report": ReportAgent,
    "nutrition": NutritionAgent,
    "medication": MedicationAgent,
    "lifestyle": LifestyleAgent,
}


def classify_intent(message: str) -> dict:
    try:
        response = LLMProvider().chat(
            [
                {"role": "system", "content": "Classify the healthcare user intent. Return only JSON: {\"intents\": [...], \"primary_intent\": \"...\"}."},
                {"role": "user", "content": message},
            ]
        )
        return json.loads(response)
    except Exception:
        lowered = message.lower()
        if "medication" in lowered or "medicine" in lowered or "side effect" in lowered:
            return {"intents": ["medication"], "primary_intent": "medication"}
        if "food" in lowered or "diet" in lowered or "meal" in lowered:
            return {"intents": ["nutrition"], "primary_intent": "nutrition"}
        if "report" in lowered or "lab" in lowered:
            return {"intents": ["report"], "primary_intent": "report"}
        if "exercise" in lowered or "sleep" in lowered or "weight" in lowered:
            return {"intents": ["lifestyle"], "primary_intent": "lifestyle"}
        return {"intents": ["diagnosis"], "primary_intent": "diagnosis"}


async def route_message(
    user_id: int,
    user_message: str,
    medical_profile: dict | None = None,
    medical_history: list | None = None,
    recent_predictions: list | None = None,
    recent_reports: list | None = None,
) -> dict:
    memory = summarize_if_needed(get_recent_memory(user_id))
    try:
        rag_context = Retriever().retrieve(user_message, top_k=5)
    except Exception:
        rag_context = []

    context = AgentContext(
        user_id=user_id,
        user_message=user_message,
        medical_profile=medical_profile or {},
        medical_history=medical_history or [],
        recent_predictions=recent_predictions or [],
        recent_reports=recent_reports or [],
        memory=memory,
        rag_context=rag_context,
    )
    save_conversation_turn(user_id, "user", user_message)

    if is_emergency(user_message):
        response = await EmergencyAgent().run(context)
        log_timeline_event(user_id, "emergency_detected", "Emergency keywords detected", user_message, response.metadata)
        save_conversation_turn(user_id, "assistant", response.content, response.metadata)
        return {"primary_intent": "emergency", "responses": [response.__dict__], "merged_response": response.content}

    intent = classify_intent(user_message)
    requested = [name for name in intent.get("intents", []) if name in AGENTS] or [intent.get("primary_intent", "diagnosis")]
    requested = [name for name in dict.fromkeys(requested) if name in AGENTS]
    responses: list[AgentResponse] = await asyncio.gather(*(AGENTS[name]().run(context) for name in requested))
    merged = "\n\n".join(f"{response.content}" for response in responses)
    save_conversation_turn(user_id, "assistant", merged, {"agents": requested, "primary_intent": intent.get("primary_intent")})
    return {
        "primary_intent": intent.get("primary_intent"),
        "responses": [response.__dict__ for response in responses],
        "merged_response": merged,
        "rag_context": rag_context,
    }
