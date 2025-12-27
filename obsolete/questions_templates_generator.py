import os
import json
from typing import List, Dict
from pathlib import Path

from common.ollama_helper import OllamaHelper, build_prompt  # предполагается, что код вынесен в модуль

MODEL = os.getenv('MODEL', 'qwen3:8b')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'npc_lora_dataset.jsonl')


SYSTEM_PROMPT = """
You are a data generation engine for game AI dialogue datasets.

Your task:
- Generate JSON objects that strictly follow the provided template.
- Each object represents a player intent mapped to a game Action.
- The output MUST be valid JSON only.
- Do NOT include explanations, comments, markdown, or extra text.
- Do NOT wrap the output in code fences.

JSON rules:
- Output MUST be either a single JSON object or a JSON array of objects.
- All keys must be present: base_template, count, action, motivation.
- Values must be strings, except "count" which must be an integer.
- Do not invent new fields.
- Preserve action names EXACTLY as provided by the user.
- If the action contains placeholders like <weapon>, <item>, or <upgrade>, keep them verbatim.

Content rules:
- base_template: a natural player utterance suitable for an RPG or tactical game.
- motivation: a concise explanation of the player's intent in neutral third-person form.
- count: always use the number provided by the user unless explicitly told otherwise.

Style:
- Neutral, in-game dialogue.
- No slang, no emojis, no meta-commentary.
""".strip()


USER_INPUT = json.dumps(
    {
        "template": {
            "base_template": "",
            "count": 5,
            "action": "",
            "motivation": ""
        },
        "actions": [
            "DoNothing()",
            "NotEnoughGoldForBuy()",
            "SoldOut()",
            "ShowAllGoodsForSale()",
            "ShowAllWeaponsForSale()",
            "ShowAllAmmoForSale()",
            "ShowAllWeaponUpgrades()",
            "ShowAllHealingItems()",
            "SellAmmo(<weapon>)",
            "SellWeapon(<weapon>)",
            "SellWeaponUpgrade(<upgrade>)",
            "SellHealingItem(<item>)"
        ]
    },
    ensure_ascii=False,
    indent=2
)


def parse_json_strict(text: str) -> List[Dict]:
    """
    Жёсткий парсер: либо валидный JSON, либо исключение.
    """
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON root type")


def main():
    helper = OllamaHelper(OLLAMA_HOST)

    if not helper.check_model_exists(MODEL):
        raise RuntimeError(f"Model not found in Ollama: {MODEL}")

    prompt = build_prompt(SYSTEM_PROMPT, USER_INPUT)

    response, thinking = helper.generate(MODEL, prompt)
    if response is None:
        raise RuntimeError("No response from Ollama")

    items = parse_json_strict(response)

    output_path = Path(OUTPUT_FILE)
    with output_path.open("a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(items)} entries to {output_path}")


if __name__ == "__main__":
    main()
