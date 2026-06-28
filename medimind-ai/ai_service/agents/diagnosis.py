from llm.provider import LLMProvider

from .base import AgentContext, AgentResponse, BaseAgent


class DiagnosisAgent(BaseAgent):
    async def run(self, context: AgentContext) -> AgentResponse:
        prompt = f"""
User symptoms: {context.user_message}
Medical history: {context.medical_history}
ML Risk Predictions: {context.recent_predictions}
Medical knowledge: {context.rag_context}

Provide: possible conditions to discuss with doctor, related risk factors, when to seek immediate care.
Always end with: "Please consult a qualified healthcare provider for diagnosis."
"""
        response = LLMProvider().chat(
            [
                {"role": "system", "content": "You are a diagnostic support AI. You do NOT diagnose. Educational information only."},
                {"role": "user", "content": prompt},
            ]
        )
        return AgentResponse("diagnosis", response)
