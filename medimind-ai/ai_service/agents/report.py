import asyncio

from llm.provider import LLMProvider

from .base import AgentContext, AgentResponse, BaseAgent


class ReportAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        response = await asyncio.to_thread(
            LLMProvider().chat,
            [
                {"role": "system", "content": "Explain medical report summaries and abnormal values in plain patient language. Do not diagnose."},
                {"role": "user", "content": f"Question: {context.user_message}\nRecent reports: {context.recent_reports}"},
            ]
        )
        return AgentResponse("report", response)

