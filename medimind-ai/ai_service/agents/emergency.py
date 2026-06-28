from .base import AgentContext, AgentResponse, BaseAgent


EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "loss of consciousness",
    "severe bleeding",
    "stroke symptoms",
    "face drooping",
    "arm weakness",
    "slurred speech",
    "suicidal",
    "overdose",
]


def is_emergency(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)


class EmergencyAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        return AgentResponse(
            agent_name="emergency",
            content=(
                "This sounds like a medical emergency. Call emergency services immediately. "
                "If it is safe, stay with the person, keep them still and comfortable, and follow dispatcher instructions. "
                "Do not delay emergency care to continue chatting."
            ),
            metadata={"emergency_detected": True},
        )
