import time
import os
from typing import Iterable

from .groq_client import GroqClient
from .openrouter_client import OpenRouterClient


class LLMProvider:
    def __init__(self):
        self.groq = GroqClient()
        self.openrouter = OpenRouterClient()

    def chat(self, messages: Iterable[dict], model: str = "groq") -> str:
        messages = list(messages)
        system_prompt = "\n".join(message["content"] for message in messages if message.get("role") == "system") or "You are MediMind AI."
        user_message = "\n".join(message["content"] for message in messages if message.get("role") == "user")
        if os.environ.get("DISABLE_LLM", "False").lower() == "true":
            return (
                "This local preview response is educational only. Your information was reviewed against the available "
                "risk factors, and the safest next step is to discuss persistent symptoms, abnormal readings, or major "
                "concerns with a qualified healthcare provider."
            )
        clients = [self.groq, self.openrouter] if model == "groq" else [self.openrouter, self.groq]
        last_error = None
        for client in clients:
            for attempt in range(3):
                try:
                    return client.complete(system_prompt, user_message)
                except Exception as exc:
                    last_error = exc
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"All LLM providers failed: {last_error}")
