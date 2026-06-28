from dataclasses import dataclass, field


@dataclass
class AgentContext:
    user_id: int
    user_message: str
    medical_profile: dict
    medical_history: list
    recent_predictions: list
    recent_reports: list
    memory: list
    rag_context: list


@dataclass
class AgentResponse:
    agent_name: str
    content: str
    metadata: dict = field(default_factory=dict)


class BaseAgent:
    async def run(self, context: AgentContext) -> AgentResponse:
        raise NotImplementedError
