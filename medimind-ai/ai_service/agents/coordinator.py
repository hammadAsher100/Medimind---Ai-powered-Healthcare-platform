import asyncio
import json
import logging

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

logger = logging.getLogger(__name__)

AGENTS = {
    "diagnosis": DiagnosisAgent,
    "report": ReportAgent,
    "nutrition": NutritionAgent,
    "medication": MedicationAgent,
    "lifestyle": LifestyleAgent,
}


def classify_intent(message: str) -> dict:
    """Classify the user message into healthcare intents.
    Uses LLM when available, falls back to keyword matching."""
    try:
        response = LLMProvider().chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the healthcare user intent into one or more categories. "
                        "Valid categories: diagnosis, report, nutrition, medication, lifestyle. "
                        'Return only JSON: {"intents": [...], "primary_intent": "..."}.'
                    ),
                },
                {"role": "user", "content": message},
            ]
        )
        # Try to parse JSON from the response
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(cleaned)
        # Validate the result structure
        if "intents" in result and "primary_intent" in result:
            return result
        raise ValueError("Invalid response structure")
    except Exception as exc:
        logger.debug("LLM intent classification failed, using keyword fallback: %s", exc)
        return _keyword_classify(message)


def _keyword_classify(message: str) -> dict:
    """Keyword-based fallback for intent classification."""
    lowered = message.lower()
    if any(kw in lowered for kw in ("medication", "medicine", "drug", "side effect", "pill", "tablet", "dose")):
        return {"intents": ["medication"], "primary_intent": "medication"}
    if any(kw in lowered for kw in ("food", "diet", "meal", "eat", "nutrition", "calorie", "protein")):
        return {"intents": ["nutrition"], "primary_intent": "nutrition"}
    if any(kw in lowered for kw in ("report", "lab", "test result", "blood test", "scan")):
        return {"intents": ["report"], "primary_intent": "report"}
    if any(kw in lowered for kw in ("exercise", "sleep", "weight", "workout", "walk", "run", "lifestyle", "stress")):
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
    # Retrieve memory safely (non-blocking)
    try:
        memory = await asyncio.to_thread(get_recent_memory, user_id)
        memory = summarize_if_needed(memory)
    except Exception:
        memory = []

    # Retrieve RAG context safely
    try:
        rag_context = await asyncio.to_thread(Retriever().retrieve, user_message, 5)
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

    # Save user turn (non-blocking)
    try:
        await asyncio.to_thread(save_conversation_turn, user_id, "user", user_message)
    except Exception:
        pass

    # Emergency detection
    if is_emergency(user_message):
        response = await EmergencyAgent().run(context)
        try:
            await asyncio.to_thread(
                log_timeline_event, user_id, "emergency_detected",
                "Emergency keywords detected", user_message, response.metadata,
            )
            await asyncio.to_thread(
                save_conversation_turn, user_id, "assistant", response.content, response.metadata,
            )
        except Exception:
            pass
        return {"primary_intent": "emergency", "responses": [response.__dict__], "merged_response": response.content}

    # Classify intent (run in thread to avoid blocking)
    try:
        intent = await asyncio.to_thread(classify_intent, user_message)
    except Exception:
        intent = _keyword_classify(user_message)

    requested = [name for name in intent.get("intents", []) if name in AGENTS] or [intent.get("primary_intent", "diagnosis")]
    requested = [name for name in dict.fromkeys(requested) if name in AGENTS]

    if not requested:
        requested = ["diagnosis"]

    # Run agents concurrently
    responses: list[AgentResponse] = await asyncio.gather(*(AGENTS[name]().run(context) for name in requested))
    merged = "\n\n".join(f"{response.content}" for response in responses)

    # Save assistant response (non-blocking)
    try:
        await asyncio.to_thread(
            save_conversation_turn, user_id, "assistant", merged,
            {"agents": requested, "primary_intent": intent.get("primary_intent")},
        )
    except Exception:
        pass

    return {
        "primary_intent": intent.get("primary_intent"),
        "responses": [response.__dict__ for response in responses],
        "merged_response": merged,
        "rag_context": rag_context,
    }
