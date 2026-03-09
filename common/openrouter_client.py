import os
import requests
from typing import Dict, List, Optional, Tuple

MODEL = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-r1:free')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'npc_lora_dataset.jsonl')

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(system_text: str, user_json: str) -> str:
    return f"SYSTEM:\n{system_text}\n\nUSER:\n{user_json}"


class OllamaHelper:
    def __init__(self, host: str = None):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    def check_model_exists(self, model: str) -> bool:
        return True

    def generate(self, model: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.9,
            "top_p": 0.95,
        }

        try:
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]

            response_text = msg.get("content")

            thinking_text = msg.get("reasoning_content")

            return response_text, thinking_text

        except Exception as e:
            print(f"[ERR] openrouter generate: {e}")
            return None, None
