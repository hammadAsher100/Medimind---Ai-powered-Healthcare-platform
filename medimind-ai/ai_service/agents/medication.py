from llm.provider import LLMProvider

from .base import AgentContext, AgentResponse, BaseAgent


class MedicationAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        response = LLMProvider().chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You explain medications, common side effects, and interaction questions in educational language. "
                        "You must never prescribe medications or suggest specific dosages. This information is educational only. "
                        "Always end with: Follow your doctor's prescription."
                    ),
                },
                {"role": "user", "content": f"Question: {context.user_message}\nProfile: {context.medical_profile}"},
            ]
        )
        return AgentResponse("medication", response)
