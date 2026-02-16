import json
import os
from typing import List, Tuple, Dict
from unidecode import unidecode

from common.helpers import (
    calculate_dataset_params,
    extract_nsloctext_value,
    save_dict_records_to_jsonl,
    parse_action_signature,
    replace_unicode
)
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL
from common.template_gen_components import env, build_action_template_params, render_template

DATASET_SIZE_PER_ACTION = int(os.getenv('DATASET_SIZE_PER_ACTION', 4000))
MAX_QUERIES_PER_ACTION_CHUNK = 50

sp_template_f_path = os.getenv('SP_TEMPLATE_F_PATH', '')
actions_f_path = os.getenv('ACTIONS_F_PATH', '')
usr_roles_f_path = os.getenv('USR_ROLES_F_PATH', '')
npc_desc_f_path = os.getenv('NPC_DESC_F_PATH', '')
npc_name = os.getenv('NPC_NAME', '')


def get_roles() -> List[dict]:
    with open(f"{usr_roles_f_path}", "r", encoding="utf-8") as f:
        roles: List[dict] = json.load(f)
        return roles

def get_system_prompt_template() -> str:
    with open(sp_template_f_path, "r", encoding="utf-8") as f:
        return f.read()

def get_npc_data():
    with open(f"{npc_desc_f_path}", "r", encoding="utf-8") as f:
        npc_data = json.load(f)
        description = extract_nsloctext_value(npc_data['Description'])
        return description

SYSTEM_PROMPT_TEMPLATE = get_system_prompt_template()

def build_system_prompt(
    request_example: str,
    player_role: Dict,
    npc_role: str,
    action: str,
    target_parameters: str,
    num_requests: int
) -> str:
    sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)
    params = {
        "request_template": request_example,
        "action": action,
        "target_parameters": target_parameters,
        "player_role_json": json.dumps(player_role, indent=2, ensure_ascii=False),
        "npc_role_json": npc_role,
        "num_requests": num_requests
    }
    rendered_template = sp_template.render(params)
    return rendered_template

def calculate_roles_and_request_amount(
        current_actions_count: int,
        params_combination_count: int,
        max_roles_count: int,
) -> Tuple[int, int]:
    solutions = calculate_dataset_params(
        dataset_size=DATASET_SIZE_PER_ACTION,

        actions=current_actions_count,
        params=params_combination_count,

        roles_min=1,
        roles_max=max_roles_count,
        queries_min=5,
        queries_max=MAX_QUERIES_PER_ACTION_CHUNK,
        tolerance=0.05
    )

    if solutions:
        target_roles_count = solutions[0]["roles"]
        target_queries_count = solutions[0]["queries"]
    else:
        target_roles_count = max_roles_count
        target_queries_count = MAX_QUERIES_PER_ACTION_CHUNK
        print(f"=== Warning. Didn't calculate proper roles and requests ===")

    return target_roles_count, target_queries_count

def main():
    actions_count = {}

    roles = get_roles()

    with open(actions_f_path) as f:
        for npc_data in json.load(f):
            if npc_data['Name'] == npc_name:
                actions_template_data = npc_data['ActionData']
                npc_description = extract_nsloctext_value(npc_data['Description'])

                for action_template_data in actions_template_data:
                    action_name, arg_names = parse_action_signature(action_template_data['ActionTemplate'])
                    if action_name not in actions_count:
                        actions_count[action_name] = 0
                    actions_count[action_name] += 1

                for action_template_data in actions_template_data:
                    requests_per_action = []

                    action_name, arg_names = parse_action_signature(action_template_data['ActionTemplate'])

                    t_str = json.dumps(action_template_data)
                    context = build_action_template_params(action_template_data)
                    request_combinations = render_template(t_str, context)

                    current_actions_count = actions_count[action_name]
                    request_combination_count = len(request_combinations)
                    max_roles_count = len(roles)

                    roles_amount, requests_amount = calculate_roles_and_request_amount(
                        current_actions_count=current_actions_count,
                        params_combination_count=request_combination_count,
                        max_roles_count=max_roles_count,
                    )

                    roles = roles[:roles_amount]

                    print(
                        f'=== '
                        f'roles: {roles_amount} '
                        f'requests_amount: {requests_amount} '
                        f'params_combination_amount: {request_combination_count} '
                        f'current_actions_amount: {current_actions_count} '
                        f'==='
                    )
                    print(f'=== Total: {roles_amount * requests_amount * request_combination_count * current_actions_count} ===')

                    for role in roles:
                        for rc in request_combinations:
                            if rc:
                                request_json = json.loads(rc)
                                action_data_str = request_json['ActionTemplate']
                                action_name, args = parse_action_signature(action_data_str)

                                npc_state = request_json['NpcStateTemplate']
                                usr_state = request_json['UsrStateTemplate']

                                request_template = request_json['RequestTemplate']
                                print()
                                print(f'--- Request Template: {request_template} ---')
                                print(f'--- Target NPC action: {action_data_str} ---')

                                prompt = build_system_prompt(
                                    request_example=request_json['RequestTemplate'],
                                    action=action_data_str,
                                    target_parameters=str.join(', ', args),
                                    player_role=role,
                                    npc_role=npc_description,
                                    num_requests=requests_amount
                                )
                                MAX_ATTEMPTS = 5
                                attempt_count = 0

                                while attempt_count < MAX_ATTEMPTS:
                                    try:
                                        helper = OllamaHelper(OLLAMA_HOST)
                                        requests_str, think = helper.generate(MODEL, prompt)
                                        result = replace_unicode(requests_str)
                                        requests = set(json.loads(result))
                                        if len(requests) < requests_amount:
                                            print(f'    === WARNING Generated amount requests is less than necessary: {len(requests)} < {requests_amount}')
                                        for request in requests:
                                            is_ok = True
                                            for target_parameter in args:
                                                if target_parameter.lower() not in request.lower():
                                                    is_ok = False
                                                    break
                                            if not is_ok:
                                                print(f'    === WARNING Request: {request} ===')
                                            else:
                                                print(f'    === Request: {request} ===')

                                            args_dict = {}
                                            for i in range(len(arg_names)):
                                                args_dict[arg_names[i]] = args[i]

                                            usr_request = {
                                                "request": request,
                                                "usr_state": usr_state,
                                                "npc_state": npc_state,
                                            }

                                            npc_valid_action = {
                                                "emotion": "",
                                                "answer": "",
                                                "think": "",
                                                "action": {
                                                    "name": action_name,
                                                    "parameters": args_dict
                                                }
                                            }

                                            request_per_action = {
                                                'usr_request': usr_request,
                                                'npc_valid_action': npc_valid_action,
                                                'player_role': role
                                            }

                                            requests_per_action.append(request_per_action)
                                        break
                                    except Exception as e:
                                        attempt_count += 1
                                        print(f'Error on LLM generation: {e}')
                    target_dir = f'output_data/{npc_name}/0_generate_usr_requests'
                    target_fname = f'{action_name}.jsonl'
                    save_dict_records_to_jsonl(
                        records=list(requests_per_action),
                        output_file=target_fname,
                        folder_path=target_dir,
                        append=True
                    )
                    print()
                    print(f'=== Action: {action_name} Requests: {len(requests_per_action)} ===')
                    print(f'=== Saved to: {target_dir}/{target_fname} ===')
                    print()
                break


if __name__ == "__main__":
    main()
