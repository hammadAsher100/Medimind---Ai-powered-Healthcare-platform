import time
import os
from typing import Iterable

from .groq_client import GroqClient
from .openrouter_client import OpenRouterClient


def _preview_response(system_prompt: str, user_message: str) -> str:
    prompt = f"{system_prompt}\n{user_message}".lower()
    original = user_message.strip()

    if "medication" in prompt or "side effect" in prompt or "prescription" in prompt:
        return (
            "### Medication Education\n"
            "I can help you understand what a medicine is commonly used for, typical side effects to watch for, "
            "and questions to ask your clinician.\n\n"
            "- Do not stop, start, or change a medication without your doctor's guidance.\n"
            "- Watch for allergic reactions such as rash, swelling, breathing difficulty, or severe dizziness.\n"
            "- Tell your doctor or pharmacist about all medicines, supplements, and medical conditions so they can check interactions.\n\n"
            f"Your question/context: {original[:500]}\n\n"
            "I must never prescribe medications or suggest specific dosages. This information is educational only. "
            "Follow your doctor's prescription."
        )

    if "nutrition" in prompt or "diet" in prompt or "meal" in prompt or "food" in prompt:
        return (
            "### Nutrition Guidance\n"
            "A balanced plan would focus on steady blood sugar, heart-friendly choices, and meals that are easy to repeat.\n\n"
            "- Build each meal around vegetables, lean protein, and high-fiber carbohydrates.\n"
            "- Prefer whole grains, beans, lentils, fruit, nuts, yogurt, fish, and olive-oil based fats when suitable.\n"
            "- Limit sugary drinks, heavily fried foods, and large portions of refined carbohydrates.\n"
            "- If you have allergies or kidney/heart restrictions, adjust protein, sodium, and potassium with a clinician or dietitian.\n\n"
            "A simple day could be oatmeal with nuts, grilled chicken or lentils with salad, and fish or beans with vegetables. "
            "This is educational guidance, not a diagnosis or prescription."
        )

    if "lifestyle" in prompt or "exercise" in prompt or "sleep" in prompt or "hydration" in prompt or "weight" in prompt:
        return (
            "### Lifestyle Plan\n"
            "Based on the information provided, the safest general approach is gradual and consistent.\n\n"
            "- Aim for regular walking or low-impact cardio most days if your doctor has cleared exercise.\n"
            "- Add light strength training 2 to 3 days per week to support metabolism and joint health.\n"
            "- Keep sleep consistent, reduce late caffeine, and target a calming bedtime routine.\n"
            "- Hydrate regularly and watch for dizziness, chest pain, or unusual shortness of breath during activity.\n\n"
            "If symptoms appear during exercise, stop and seek medical advice. This is educational support only."
        )

    if "report" in prompt or "lab" in prompt or "cholesterol" in prompt or "hemoglobin" in prompt or "hba1c" in prompt:
        return (
            "### Report Explanation\n"
            "I can help translate lab-report language into plain English.\n\n"
            "- Values marked high or low should be interpreted with your age, symptoms, medicines, and medical history.\n"
            "- Cholesterol, LDL, HDL, triglycerides, glucose, HbA1c, hemoglobin, creatinine, TSH, WBC, RBC, and platelets are common values to review.\n"
            "- A single abnormal value does not always mean a disease, but persistent or severe abnormalities deserve follow-up.\n\n"
            "Upload the report or paste the key values, and I can help organize what to ask your doctor."
        )

    if "diagnostic" in prompt or "symptom" in prompt or "condition" in prompt or "risk factor" in prompt:
        return (
            "### Symptom Education\n"
            "I cannot diagnose you, but I can help you think through what to discuss with a healthcare provider.\n\n"
            "- Note when symptoms started, what makes them better or worse, and whether they are getting worse.\n"
            "- Track fever, pain severity, breathing changes, weakness, dizziness, vomiting, or new neurological symptoms.\n"
            "- Seek urgent care for chest pain, trouble breathing, fainting, severe bleeding, stroke-like symptoms, or rapidly worsening symptoms.\n\n"
            "Please consult a qualified healthcare provider for diagnosis."
        )

    return (
        "### MediMind Guidance\n"
        "I reviewed your message and can help with symptom education, report explanation, nutrition, lifestyle planning, medication education, "
        "or risk-assessment support.\n\n"
        "- Share the main concern, when it started, and any relevant readings or report values.\n"
        "- Mention existing conditions, allergies, medicines, and recent predictions if available.\n"
        "- Use emergency services immediately for chest pain, difficulty breathing, severe bleeding, loss of consciousness, or stroke-like symptoms.\n\n"
        "This is educational support only and does not replace care from a qualified healthcare provider."
    )


class LLMProvider:
    def __init__(self):
        self.groq = GroqClient()
        self.openrouter = OpenRouterClient()

    def chat(self, messages: Iterable[dict], model: str = "groq") -> str:
        messages = list(messages)
        system_prompt = "\n".join(message["content"] for message in messages if message.get("role") == "system") or "You are MediMind AI."
        user_message = "\n".join(message["content"] for message in messages if message.get("role") == "user")
        if os.environ.get("DISABLE_LLM", "False").lower() == "true":
            return _preview_response(system_prompt, user_message)
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
