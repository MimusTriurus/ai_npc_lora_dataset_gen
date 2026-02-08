import json
import os
import re
from collections import defaultdict

from common.helpers import is_env_var_true
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL

def extract_nsloctext_value(text: str) -> str:
    match = re.search(r'NSLOCTEXT\([^,]+,\s*[^,]+,\s*\"(.*)\"\)', text)
    if not match:
        return text

    value = match.group(1)

    value = value.encode('utf-8').decode('unicode_escape')

    return value

def build_system_prompt(action_data):
    actions = {}

    param_groups = defaultdict(set)

    for action in action_data['ActionData']:
        raw_name = action["ActionTemplate"]

        clean_name = re.sub(r"\(\s*\{\{.*?\}\}\s*\)", "", raw_name)

        param_keys = list(action["Parameters"].keys())
        actions[clean_name] = param_keys

        for key, values in action["Parameters"].items():
            param_groups[key].update(values)

    lines = []

    for action_name, param_keys in actions.items():
        lines.append(f"{action_name}")
        lines.append(f"   parameters: {json.dumps([f'<{p}>' for p in param_keys])} where")

        for p in param_keys:
            lines.append(f"      <{p}> is one of AllowedParameters_{p}")

        lines.append("")

    for key, values in param_groups.items():
        lines.append(f"AllowedParameters_{key}")
        for v in sorted(values):
            lines.append(f"- {v}")
        lines.append("")

    return "\n".join(lines)

def generate_action_description(npc_data: dict):# -> str:
    if not npc_data:
        return ""

    for action in npc_data["ActionData"]:
        sp = f'''
You are an Intent Analyzer for a game dialogue system.
User is interacting with the following NPC:
{npc_data['Description']}

Your task:
- Receive a JSON object describing an NPC action.
- Analyze the fields:
  - ActionName - describes the NPC's action in response to a user request (RequestTemplate).
  - Parameters
  - RequestTemplate - User's request to the NPC
  - UsrStateTemplate
  - NpcStateTemplate

Interpretation guidelines:
- Understand what the user is trying to achieve based on the RequestTemplate.
- Treat random functions such as rand_range() as placeholders.
- Your output must be a concise description of the user's intention in plain English.

Rules:
1. Describe only the intention of the user (what they want to achieve).
2. If the action implies conditions (gold, amount of goods, etc.), mention them abstractly:
   - "the user wants to buy an item if they have enough gold"
   - "the user wants to sell an item if the merchant has stock"
3. Do not mention parameters of the action in you response.

Action for analysis:
{json.dumps(action, indent=2)}
        '''

        helper = OllamaHelper(OLLAMA_HOST)
        MODEL = 'qwen3:8b'
        action_description, think = helper.generate(MODEL, sp)
        action['Description'] = action_description

if __name__ == "__main__":
    NPC_DESC_F_PATH = os.getenv("NPC_DESC_F_PATH")

    with open(NPC_DESC_F_PATH, "r", encoding="utf-8") as f:
        npcs_desc = json.load(f)
        for npc_data in npcs_desc:
            desc = npc_data["Description"]
            npc_data['Description'] = extract_nsloctext_value(npc_data['Description'])
            if is_env_var_true('GENERATE_ACTION_DESC'):
                generate_action_description(npc_data)
            prompt = build_system_prompt(npc_data)
            print(prompt)