import json
import math
import os
import random
import sys
from typing import Dict, List, Set
import re
import itertools
from jinja2 import Environment
from unidecode import unidecode
from common.helpers import calculate_dataset_params, save_dict_records_to_jsonl
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL

DATASET_SIZE_PER_ACTION = int(os.getenv('DATASET_SIZE_PER_ACTION', 4000))
sp_template_f_path = os.getenv('SP_TEMPLATE_F_PATH', '')
actions_f_path = os.getenv('ACTIONS_F_PATH', '')
usr_roles_f_path = os.getenv('USR_ROLES_F_PATH', '')
npc_desc_f_path = os.getenv('NPC_DESC_F_PATH', '')

def rand_range(a, b):
    return random.randint(a, b)

env = Environment()
env.globals["rand_range"] = rand_range

SYSTEM_PROMPT_TEMPLATE = ""
with open(sp_template_f_path, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT_TEMPLATE += f.read()

def build_system_prompt(
    action: Dict,
    player_role: Dict,
    npc_role: Dict,
    target_parameters: str,
    num_requests: int
) -> str:
    sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)
    params = {
        "request_template": action['RequestTemplate'],
        "target_parameters": target_parameters,
        "player_role_json": json.dumps(player_role, indent=2, ensure_ascii=False),
        "npc_role_json": json.dumps(npc_role, indent=2, ensure_ascii=False),
        "num_requests": num_requests
    }
    rendered_template = sp_template.render(params)
    return rendered_template

actions: List[dict] = []
roles = []

actions_count = {}

with open(f"{actions_f_path}", "r", encoding="utf-8") as f:
    actions: List[dict] = json.load(f)
    for action in actions:
        if action['ActionName'] not in actions_count:
            actions_count[action['ActionName']] = 0
        actions_count[action['ActionName']] += 1

# region load user's roles
with open(f"{usr_roles_f_path}", "r", encoding="utf-8") as f:
    roles: List[dict] = json.load(f)

roles_available = len(roles)
# endregion

# region load npc's role
npc_role = {}
with open(f"{npc_desc_f_path}", "r", encoding="utf-8") as f:
    npc_role.update(json.load(f))
# endregion

for action in actions:
    user_requests = []

    action_name = action['ActionName']
    usr_state_template = action['UsrStateTemplate']
    npc_state_template = action['NpcStateTemplate']
    action_params = action.get('Parameters', {})

    matches = re.findall(r"\{\{\s*(.*?)\s*\}\}", action['RequestTemplate'])

    r: List = []
    for match in matches:
        pairs = []
        for arg in action_params.get(match, []):
            kv = ( match, arg )
            pairs.append(kv)
        if pairs:
            r.append( pairs )

    combinations = list(itertools.product(*r))

    request_combinations_count = len(combinations)
    dataset_size_per_request_combinations = math.ceil(DATASET_SIZE_PER_ACTION / request_combinations_count)

    current_actions_count = actions_count[action_name]

# region calculate dataset chunk parameters
    solutions = calculate_dataset_params(
        dataset_size=DATASET_SIZE_PER_ACTION,

        actions=current_actions_count,
        params=len(combinations),

        roles_min=5,
        roles_max=roles_available,
        queries_min=5,
        queries_max=50,
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

    print(f'*** Current action {action_name}. Total: {target_queries_count * target_roles_count * len(combinations)}')
    print(f'Roles: {target_roles_count} Requests: {target_queries_count} Params: {len(combinations)}')

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
            requests_str, think = helper.generate(MODEL, prompt)

            try:
                requests: Set[str] = set(json.loads(requests_str))
                for request in requests:
                    usr_state = env.from_string(usr_state_template).render(**params)
                    npc_state = env.from_string(npc_state_template).render(**params)

                    result = unidecode(request)
                    result = result.replace('--', '-')
                    request_with_states = {
                        "request": result,
                        "usr_state": usr_state,
                        "npc_state": npc_state,
                        "params": params,
                    }
                    user_requests.append(request_with_states)
                    print(f'{action_name} [{len(user_requests)}/{DATASET_SIZE_PER_ACTION}] {result}')
                    # region [OBSOLETE]
                    '''
                    isOk = True
                    for k, v in params.items():
                        if v not in request:
                            isOk = False
                            break
                    if isOk:
                        usr_state = env.from_string(usr_state_template).render(**params)
                        npc_state = env.from_string(npc_state_template).render(**params)

                        result = unidecode(request)
                        result = result.replace('--', ' - ')
                        request_with_states = {
                            "request": result,
                            "usr_state": usr_state,
                            "npc_state": npc_state,
                            "params": params,
                        }
                        user_requests.append(request_with_states)
                        print(f'--- {result}')
                    else:
                        print(f'*** Error! Args {params} not found in the {request}')
                    '''
                    # endregion
            except Exception as e:
                print(e)

    save_dict_records_to_jsonl(user_requests, f'{action_name}.jsonl', append=True)

print('====')
