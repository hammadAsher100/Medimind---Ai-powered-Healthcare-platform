import asyncio

from llm.provider import LLMProvider

from .base import AgentContext, AgentResponse, BaseAgent


class LifestyleAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        response = await asyncio.to_thread(
            LLMProvider().chat,
            [
                {"role": "system", "content": "Give educational lifestyle support for exercise, sleep, hydration, and weight goals based on the health profile."},
                {"role": "user", "content": f"Message: {context.user_message}\nProfile: {context.medical_profile}\nPredictions: {context.recent_predictions}"},
            ]
        )
        return AgentResponse("lifestyle", response)

