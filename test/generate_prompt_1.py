import json
import math
import os
import sys
from typing import Dict, List
import re
import itertools
from jinja2 import Environment
from unidecode import unidecode
from common.helpers import calculate_dataset_params
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL, build_prompt

DATASET_SIZE_PER_ACTION = int(os.getenv('DATASET_SIZE_PER_ACTION', 4000))

SYSTEM_PROMPT_TEMPLATE = ""
with open("resources/system_prompt.j2", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT_TEMPLATE += f.read()

def build_system_prompt(
    action: Dict,
    player_role: Dict,
    npc_role: Dict,
    target_parameters: str,
    num_requests: int
) -> str:
    env = Environment()
    sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)
    params = {
        "action": action['ActionName'],
        "intention": action['RequestTemplate'],
        "target_parameters": target_parameters,
        "player_role_json": json.dumps(player_role, indent=2, ensure_ascii=False),
        "npc_role_json": json.dumps(npc_role, indent=2, ensure_ascii=False),
        "num_requests": num_requests
    }
    rendered_template = sp_template.render(params)
    return rendered_template

actions = []
roles = []

action_object = {
    "ActionName": "SellItem",
    "Parameters":
    {
        "weapon": ["pistol", "rifle", "shotgun", "revolver", "sniper_rifle", "rocket_launcher"],
        "name": ["Alex", "Jake"]
    },
    #"RequestTemplate": "Could you sell me the {{ weapon }} for defence from monsters"
    "RequestTemplate": "Hi! My name is {{ name }}. Could you sell me the {{ weapon }} for defence from monsters"
}

actions.append(action_object)

# region load user's roles
with open("resources/user_roles.json", "r", encoding="utf-8") as f:
    roles: List[dict] = json.load(f)

roles_available = len(roles)
# endregion

# region load npc's role
npc_role = {}
with open("resources/npc_trader/npc_description.json", "r", encoding="utf-8") as f:
    npc_role.update(json.load(f))
# endregion

for action in actions:
    user_requests = set()

    matches = re.findall(r"\{\{\s*(.*?)\s*\}\}", action_object['RequestTemplate'])

    r: List = []
    for match in matches:
        args = action_object['Parameters'].get(match, [])
        pairs = []
        for arg in args:
            kv = ( match, arg )
            pairs.append(kv)
        if pairs:
            r.append( pairs )

    combinations = list(itertools.product(*r))

    request_combinations_count = len(combinations)
    dataset_size_per_request_combinations = math.ceil(DATASET_SIZE_PER_ACTION / request_combinations_count)

# region calculate dataset chunk parameters
    solutions = calculate_dataset_params(
        dataset_size=DATASET_SIZE_PER_ACTION,

        actions=len(combinations),
        params=1,

        roles_min=1,
        roles_max=roles_available,
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
# endregion

    roles = roles[:target_roles_count]

    for combination in combinations:
        params = {}
        target_params = []
        for c in combination:
            k, v = c
            target_params.append(f'{k}: {v}')
            params.update({k: v})

        target_params_str = ', '.join(target_params)

        for player_role in roles:
            prompt = build_system_prompt(
                action=action,
                player_role=player_role,
                npc_role=npc_role,
                target_parameters=target_params_str,
                num_requests=target_queries_count
            )

            helper = OllamaHelper(OLLAMA_HOST)
            new_template, think = helper.generate(MODEL, prompt)

            try:
                res = set(json.loads(new_template))
                for r in res:
                    isOk = True
                    for k, v in params.items():
                        if k not in r:
                            isOk = False
                            break
                    if isOk:
                        result = unidecode(r)
                        result = result.replace('--', ' - ')
                        user_requests.add(result)
                        print(f'--- {result}')
            except Exception as e:
                print(e)

    print(f'{len(user_requests)}/{DATASET_SIZE_PER_ACTION}')
