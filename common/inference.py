import json
import os
from pathlib import Path
from typing import Dict, Type

from common.constants import DATA_DIR_NAME, DATASET_DIR_NAME, GEN_SYS_PROMPT_DIR_NAME
from common.helpers import read_dataset_file

def load_ullm_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
        return config


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        return data


def get_system_prompt(sp_path: str):
    sp = read_file(sp_path)

    system_prompt = sp

    return system_prompt

def get_grammar(grammar_path):
    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()
        return grammar

def make_ullama_config(git_commit: str, npc_name: str, flow_run_id: str, model: str, lora: str, sp: str = 'system_prompt.txt') -> dict:
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/{sp}'
    grammar_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/grammar.txt'

    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')
    llm_config = load_ullm_config(f'{llm_cfg_f_path}')
    llm_config['model'] = model
    llm_config['lora_adapter'] = lora
    llm_config['system_prompt'] = get_system_prompt(system_prompt_f_path)
    grammar_string = get_grammar(grammar_f_path)
    llm_config['grammar'] = grammar_string

    return llm_config

def list_files(folder_path: str) -> list[str]:
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    ]

def inference(git_commit: str, npc_name: str, flow_run_id: str, inference_config: dict, inference_type: Type):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    # region ENV vars
    requests_count = int(os.getenv('STEP_2_MAX_REQUESTS_COUNT', 10))
    validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation'
    # endregion

    if not os.path.isdir(validation_dataset_dir_path):
        print("No validation dataset path provided")
        exit(1)

    dataset_files = list_files(validation_dataset_dir_path)

    ullama = inference_type(inference_config)

    total_fails = 0
    total_requests = 0

    json_parse_fails: int = 0
    json_structure_fails: int = 0
    action_fails: Dict[str, int] = {}
    args_fails: Dict[str, int] = {}

    for dataset_file in dataset_files:
        file_name = Path(dataset_file).stem

        action_fails[file_name] = 0
        args_fails[file_name] = 0

        dataset_pairs = read_dataset_file(dataset_file)
        counter = 1
        dataset_pairs = dataset_pairs[:requests_count]
        requests_count = min(requests_count, len(dataset_pairs))
        for pair in dataset_pairs:
            print(f'--> {file_name} [{counter}/{requests_count}]')
            counter += 1
            total_requests += 1

            request = pair[0]
            valid_response_dict = pair[1]
            request_dict = json.loads(request)

            llm_model_f_path = inference_config.get('model', '')
            response_dict, think_block = ullama.chat(
                model=llm_model_f_path,
                system_prompt=inference_config.get('system_prompt', ''),
                user_prompt=request
            )

            response = ''

            if response_dict is None:
                json_parse_fails += 1
                total_fails += 1
                print(f"==> Error! Can't parse response: {response}")
                continue

            if 'answer' not in response_dict or 'action' not in response_dict or 'name' not in response_dict[
                'action'] or 'parameters' not in response_dict['action']:
                json_structure_fails += 1
                total_fails += 1
                print(f"==> Error!. Json has incorrect structure: {response}")
                continue

            answer = response_dict.get('answer')
            action_name = response_dict['action']['name']
            action_args = response_dict['action']['parameters']

            valid_action_name = valid_response_dict['action']['name']
            valid_action_args = valid_response_dict['action']['parameters']

            if action_name != valid_action_name:
                total_fails += 1
                action_fails[file_name] += 1
                print(
                    f'==> Error! Wrong action!\n Valid: {valid_action_name} {valid_action_args}\n Current: {action_name} {action_args}')
                print(f'CONTEXT:\n{request_dict["usr_state"]}\n{request_dict["npc_state"]}')
                continue

            if action_args != valid_action_args:
                total_fails += 1
                args_fails[file_name] += 1
                print(f'==> Error! Wrong action args!\n Valid: {valid_action_args}\n Current: {action_args}')
    print('')
    print('--- RESULTS ---')
    print(f'=== Json parse fails: {json_parse_fails}/{total_requests} ===')
    print(f'=== Json structure fails: {json_structure_fails}/{total_requests} ===')
    print(f'=== Action fails ===')
    print(action_fails)
    print(f'=== Args fails ===')
    print(args_fails)
    print(f'--- Total fails: {total_fails}/{total_requests} ---')

    print('=== end ===')

    metrics = {
        "total_requests": total_requests,

        "total_fails": total_fails,

        "json_parse_fails": json_parse_fails,
        "json_structure_fails": json_structure_fails,
        "total_action_fails": sum(action_fails.values()),
        "total_args_fails": sum(args_fails.values()),

        "fails_per_action": action_fails,
        "fails_per_action_args": args_fails,
    }

    return metrics