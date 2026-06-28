from llm.provider import LLMProvider

from .base import AgentContext, AgentResponse, BaseAgent


class NutritionAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        response = LLMProvider().chat(
            [
                {"role": "system", "content": "Create practical nutrition guidance tailored to allergies, medical profile, and risk predictions. Do not diagnose."},
                {
                    "role": "user",
                    "content": f"Message: {context.user_message}\nProfile: {context.medical_profile}\nPredictions: {context.recent_predictions}\nHistory: {context.medical_history}",
                },
            ]
        )
        return AgentResponse("nutrition", response)
