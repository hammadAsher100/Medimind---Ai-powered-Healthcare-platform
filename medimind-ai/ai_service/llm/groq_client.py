import os

from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqClient:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        if not self.client:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
