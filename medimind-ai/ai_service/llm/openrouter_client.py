import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct"


class OpenRouterClient:
    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MediMind AI"),
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
