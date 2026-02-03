import json
import sys
from typing import Dict

from unidecode import unidecode

from common.helpers import calculate_dataset_params
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL

SYSTEM_PROMPT_TEMPLATE = """You are a dialogue data generator for a game AI system.
Your task is to generate multiple player requests that will cause an NPC to invoke the following action with a SPECIFIC parameter.
{action}

Target action parameter:
{target_parameter}

The player who speaks these requests has the following role:
{player_role_json}

The player's intention:
{intention}

Generation rules:
- Generate EXACTLY {num_requests} different player requests.
- The target_parameter has to be mentioned in the player's request.
- Each request must clearly and unambiguously imply the SAME target action parameter.
- The parameter must NOT be mentioned explicitly or by its internal name.
- Requests must be natural, immersive, and consistent with the player's speech_style and motivation.
- The intent to buy must be obvious.
- Avoid any ambiguity with other parameters.
- Do NOT mention gold, prices, inventory counts, or game mechanics.
- Vary wording, sentence structure, and phrasing between requests.
- Keep each request to 1–2 sentences.

Output format:
A JSON array of strings.
No extra fields. No explanations.
"""

def build_system_prompt(
    action: Dict,
    role: Dict,
    target_parameter: str,
    num_requests: int
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        action=action['ActionName'],
        intention=action['Description'],
        target_parameter=target_parameter,
        player_role_json=json.dumps(role, indent=2, ensure_ascii=False),
        num_requests=num_requests
    )

actions = []
roles = []

action_object = {
    "ActionName": "SellItem",
    "Parameters": [
        "pistol",
        "rifle",
        "shotgun",
        "revolver",
        "sniper_rifle",
        "rocket_launcher"
    ],
    "Description": "The user asks to sell him a weapon, he has enough gold and the amount of goods from the merchant is > 0"
}

actions.append(action_object)

role_object = {
    "id": "rookie_survivor",
    "name": "Rookie Survivor",
    "description": "A newcomer trying to stay alive in a dangerous zone.",
    "speech_style": "nervous, hesitant, polite",
    "motivation": "needs basic weapons and healing items"
}

roles.append(role_object)

solutions = calculate_dataset_params(
    actions=len(actions),
    params=len(actions[0]['Parameters']),

    roles_min=1,
    roles_max=60,
    queries_min=1,
    queries_max=100,
    tolerance=0.05
)

target_roles_count = 0
target_queries_count = 0

if solutions:
    target_roles_count = solutions[0]["roles"]
    target_queries_count = solutions[0]["queries"]

if not target_roles_count or not target_queries_count:
    exit(1)

roles = roles[:target_roles_count]

for role in roles:
    for action in actions:
        for parameter in action_object["Parameters"]:
            prompt = build_system_prompt(
                action=action,
                role=role,
                target_parameter=parameter,
                num_requests=target_queries_count
            )

            helper = OllamaHelper(OLLAMA_HOST)

            new_template, think = helper.generate(MODEL, prompt)

            res = set(json.loads(new_template))

            for r in res:
                if parameter not in r:
                    print(f'!!! error: {r}')
                else:
                    result = unidecode(r)
                    result = result.replace('--', ' - ')
                    print(f'--- {result}')
            break
