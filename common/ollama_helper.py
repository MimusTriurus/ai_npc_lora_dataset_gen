import os

import ollama
from typing import Dict, List, Optional, Tuple

import json

MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b-instruct')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'npc_lora_dataset.jsonl')

def build_prompt(system_text: str, user_json: str) -> str:
    return f"SYSTEM:\n{system_text}\n\nUSER:\n{user_json}"

class OllamaHelper:
    def __init__(self, base_config: dict):
        host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
        if not base_config:
            with open('resources/ollama_cfg.json', 'r') as f:
                base_config = json.load(f)

        self.temp: float = base_config.get('temp', 0.3)
        self.top_k: int = base_config.get('top_k', 40)
        self.top_p: float = base_config.get('top_p', 0.8)
        self.repeat_penalty: float = base_config.get('repeat_penalty', 1.2)
        self.repeat_last_n: int = base_config.get('repeat_last_n', 64)
        self.seed: int = base_config.get('seed', -1)

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
                    "temperature": self.temp,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                    "seed": self.seed,
                },
            )
            return resp.get('response', None), resp.get('thinking', None)
        except ollama.ResponseError as e:
            print(f"[ERR] ollama generate: {e}")
            return None, None
        except Exception as e:
            print(f"[ERR] unexpected: {e}")
            return None, None

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[dict]] = None,
    ) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            messages = []

            # system prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # previous dialogue
            if history:
                messages.extend(history)

            # new user message
            messages.append({"role": "user", "content": user_prompt})

            resp = self.client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={
                    "temperature": self.temp,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                    "seed": self.seed,
                },
            )

            # Ollama chat returns:
            # {"message": {"role": "assistant", "content": "..."}}
            msg = resp.get("message", {})
            answer = msg.get("content", None)
            thinking = resp.get("thinking", None)

            return json.loads(answer), thinking

        except ollama.ResponseError as e:
            print(f"[ERR] ollama chat: {e}")
            return None, None
        except Exception as e:
            print(f"[ERR] unexpected: {e}")
            return None, None