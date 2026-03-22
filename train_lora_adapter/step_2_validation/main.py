import json
import os
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict
from common.constants import *
from common.helpers import update_manifest, replace_unicode
from ullama_python.ullama import emotions, split_think_and_json, ULlamaWrapper
from prefect import task

from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL
from common.ullama_helper import ULlamaHelper


def list_files(folder_path: str) -> list[str]:
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    ]

def load_ullm_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
        return config

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
        return data

def read_dataset_file(path: str) -> List[Tuple[str, dict]]:
    dataset_pairs: List[Tuple[str, dict]] = []
    with open(path, "r", encoding="utf-8") as f:
        data_lines = f.readlines()
        for line in data_lines:
            json_data = json.loads(line)
            usr_request = json_data['messages'][1]['content']
            ai_response = json.loads(json_data['messages'][2]['content'])
            pair = (usr_request, ai_response)
            dataset_pairs.append(pair)
    return dataset_pairs

def make_system_prompt(sp_path: str):
    sp = read_file(sp_path)

    system_prompt = sp

    return system_prompt

def parse_actions_from_file(path: str):
    actions = []

    with open(path, "r", encoding="utf-8") as f:
        actions_desc = json.loads(f.read())
        for action_name, desc in actions_desc.items():
            actions.append({"name": action_name})

    return actions

def custom_build_grammar(emotions: List[str], actions: List[dict], use_thinking_mode: bool = True) -> str:
    header = r'root ::= ThinkOrNothing nl nl Response' if use_thinking_mode else r'root ::= Response'

    thinking_rules = r'''
ThinkOrNothing ::= ThinkBlock | ""
ThinkBlock ::= "<think>" ThinkText "</think>"
Sentence ::= ([^.<] | "<" [^/])* "."
ThinkText ::= Sentence | Sentence Sentence | Sentence Sentence Sentence
''' if use_thinking_mode else r''

    common_rules = r'''
nl ::= "\n"
Action ::= "{" ws "\"name\":" ws actions "," ws "\"parameters\":" ws dict "}"
Response ::= "{" ws "\"emotion\":" ws emotions "," ws "\"answer\":" ws string "," ws "\"action\":" ws Action "}"
string ::= "\"" ([^"]*) "\""
ws ::= [ \t\n]*
kv ::= string ws ":" ws string
dict ::= "{" ws "}" | "{" ws kv ("," ws kv)* ws "}"
    '''

    emotions_rule = "emotions ::= " + " | ".join([rf'"\"{e}\""' for e in emotions])
    actions_rule = "actions ::= " + " | ".join([rf'"\"{a["name"]}\""' for a in actions])

    result =  f"# GBNF Grammar\n{header}{thinking_rules}{common_rules}\n{emotions_rule}\n{actions_rule}"
    return result

@task(name="step_2_lora_validation")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = f'{flow_run_dir_path}/manifest.json'

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    # region ENV vars
    actions_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/actions_desc.json'
    requests_count = int(os.getenv('STEP_2_MAX_REQUESTS_COUNT', 10))
    use_thinking = os.getenv('STEP_2_USE_THINKING', 'false').lower() in ("1", "true", "yes", "on")
    system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'

    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')

    validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation'
    # endregion

    if not os.path.isdir(validation_dataset_dir_path):
        print("No validation dataset path provided")
        exit(1)

    dataset_files = list_files(validation_dataset_dir_path)

    ullm_config = load_ullm_config(f'{llm_cfg_f_path}')
    ullm_config['model'] = llm_model_f_path
    ullm_config['lora_adapter'] = lora_adapter_f_path
    ullm_config['system_prompt'] = make_system_prompt(system_prompt_f_path)
    actions = parse_actions_from_file(actions_f_path)
    grammar_string = custom_build_grammar(emotions, actions, use_thinking)
    ullm_config['grammar'] = grammar_string

    ullama_cfg_json_str = json.dumps(ullm_config).encode('utf-8')

    try:
        api = ULlamaWrapper()

        model = api.lib.ullama_loadModel(ullama_cfg_json_str)
        worker = api.lib.ullama_worker_make()
        if api.lib.ullama_worker_init(worker, ullama_cfg_json_str, model):
            token_buf = ctypes.create_string_buffer(512)
            api.lib.ullama_worker_run(worker)
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
                for pair in dataset_pairs[:requests_count]:
                    print(f'--> {file_name} [{counter}/{requests_count}]')
                    counter += 1
                    total_requests += 1

                    request = pair[0]
                    valid_response_dict = pair[1]
                    request_dict = json.loads(request)

                    api.lib.ullama_worker_ask(worker, request.encode('utf-8'))
                    response = ''
                    while api.lib.ullama_worker_isSpeaking(worker):
                        if api.lib.ullama_worker_getToken(worker, token_buf, 512):
                            response += token_buf.value.decode('utf-8')

                    think_block = None
                    response_dict = {}
                    try:
                        response = replace_unicode(response)
                        think_block, response_dict = split_think_and_json(response)
                    except Exception as e:
                        print(e)
                        continue

                    if response_dict is None:
                        json_parse_fails += 1
                        total_fails += 1
                        print(f"==> Error! Can't parse response: {response}")
                        continue

                    if 'answer' not in response_dict or 'action' not in response_dict or 'name' not in response_dict['action'] or 'parameters' not in response_dict['action']:
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
                        print(f'==> Error! Wrong action!\n Valid: {valid_action_name} {valid_action_args}\n Current: {action_name} {action_args}')
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
        else:
            api.lib.ullama_worker_dispose(worker)
            api.lib.ullama_freeModel(model)

        api.lib.ullama_worker_dispose(worker)
        api.lib.ullama_freeModel(model)
    except Exception as e:
        print(f"Error: {e}")

    print('=== end ===')

def process_ollama(git_commit: str, npc_name: str, flow_run_id: str):
    helper = OllamaHelper(OLLAMA_HOST)

    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = f'{flow_run_dir_path}/manifest.json'

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    requests_count = int(os.getenv('STEP_2_MAX_REQUESTS_COUNT', 10))
    system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'

    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')

    validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation'
    # endregion

    if not os.path.isdir(validation_dataset_dir_path):
        print("No validation dataset path provided")
        exit(1)

    dataset_files = list_files(validation_dataset_dir_path)

    system_prompt = make_system_prompt(system_prompt_f_path)

    try:
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
            for pair in dataset_pairs[:requests_count]:
                print(f'--> {file_name} [{counter}/{requests_count}]')
                counter += 1
                total_requests += 1

                request = pair[0]
                valid_response_dict = pair[1]
                request_dict = json.loads(request)

                response, thinking = helper.chat(
                    model=MODEL,
                    system_prompt=system_prompt,
                    user_prompt=request
                )
                response = replace_unicode(response)
                response_dict = None
                try:
                    response_dict = json.loads(response)
                except Exception as e:
                    print(e)

                if response_dict is None:
                    json_parse_fails += 1
                    total_fails += 1
                    print(f"==> Error! Can't parse response: {response}")
                    continue

                if 'answer' not in response_dict or 'action' not in response_dict or 'name' not in response_dict['action'] or 'parameters' not in response_dict['action']:
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
                    print(f'==> Error! Wrong action!\n Valid: {valid_action_name} {valid_action_args}\n Current: {action_name} {action_args}')
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
    except Exception as e:
        print(f"Error: {e}")

    print('=== end ===')

def process_ullama(git_commit: str, npc_name: str, flow_run_id: str):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = f'{flow_run_dir_path}/manifest.json'

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    # region ENV vars
    actions_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/actions_desc.json'
    requests_count = int(os.getenv('STEP_2_MAX_REQUESTS_COUNT', 10))
    use_thinking = os.getenv('STEP_2_USE_THINKING', 'false').lower() in ("1", "true", "yes", "on")
    system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'

    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')

    validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation'
    # endregion

    if not os.path.isdir(validation_dataset_dir_path):
        print("No validation dataset path provided")
        exit(1)

    dataset_files = list_files(validation_dataset_dir_path)

    ullm_config = load_ullm_config(f'{llm_cfg_f_path}')
    ullm_config['model'] = llm_model_f_path
    ullm_config['lora_adapter'] = lora_adapter_f_path
    ullm_config['system_prompt'] = make_system_prompt(system_prompt_f_path)
    actions = parse_actions_from_file(actions_f_path)
    grammar_string = custom_build_grammar(emotions, actions, use_thinking)
    ullm_config['grammar'] = grammar_string

    try:
        ullama = ULlamaHelper(ullm_config)

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
            for pair in dataset_pairs[:requests_count]:
                print(f'--> {file_name} [{counter}/{requests_count}]')
                counter += 1
                total_requests += 1

                request = pair[0]
                valid_response_dict = pair[1]
                request_dict = json.loads(request)

                response_dict, think_block = ullama.chat(
                    model=ullm_config['model'],
                    system_prompt=ullm_config['system_prompt'],
                    user_prompt=request
                )

                response = ''

                if response_dict is None:
                    json_parse_fails += 1
                    total_fails += 1
                    print(f"==> Error! Can't parse response: {response}")
                    continue

                if 'answer' not in response_dict or 'action' not in response_dict or 'name' not in response_dict['action'] or 'parameters' not in response_dict['action']:
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
                    print(f'==> Error! Wrong action!\n Valid: {valid_action_name} {valid_action_args}\n Current: {action_name} {action_args}')
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
    except Exception as e:
        print(f"Error: {e}")

    print('=== end ===')

if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'
    process_ullama(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)