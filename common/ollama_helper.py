import os

import ollama
from typing import Dict, List, Optional, Tuple

MODEL = os.getenv('MODEL', 'qwen3:4b-instruct')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'npc_lora_dataset.jsonl')

def build_prompt(system_text: str, user_json: str) -> str:
    return f"SYSTEM:\n{system_text}\n\nUSER:\n{user_json}"

class OllamaHelper:
    def __init__(self, host: str):
        self.client = ollama.Client(host=host)

    def check_model_exists(self, model: str) -> bool:
        try:
            models = self.client.list()
            names = [m['model'] for m in models['models']]
            return model in names
        except Exception as e:
            print(f"[ERR] listing models: {e}")
            return False

    def generate(self, model: str, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            resp = self.client.generate(
                model=model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "top_k": 20
                },
            )
            return resp.get('response', None), resp.get('thinking', None)
        except ollama.ResponseError as e:
            print(f"[ERR] ollama generate: {e}")
            return None, None
        except Exception as e:
            print(f"[ERR] unexpected: {e}")
            return None, None