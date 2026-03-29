import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Type
from common.constants import *
from ullama_python.ullama import emotions
from prefect import task

from common.helpers import update_manifest, parse_actions_from_file
from common.metrics_plot_generation import make_metrics_plot
from common.ollama_helper import OllamaHelper, MODEL
from common.training_results_report_generation import generate_validation_report
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


def get_system_prompt(sp_path: str):
    sp = read_file(sp_path)

    system_prompt = sp

    return system_prompt

def get_grammar(grammar_path):
    with open(grammar_path, "r", encoding="utf-8") as f:
        grammar = f.read()
        return grammar

def make_ullama_config(git_commit: str, npc_name: str, flow_run_id: str, model: str, lora: str) -> dict:
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'
    grammar_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/grammar.txt'

    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')
    llm_config = load_ullm_config(f'{llm_cfg_f_path}')
    llm_config['model'] = model
    llm_config['lora_adapter'] = lora
    llm_config['system_prompt'] = get_system_prompt(system_prompt_f_path)
    grammar_string = get_grammar(grammar_f_path)
    llm_config['grammar'] = grammar_string

    return llm_config


def inference(git_commit: str, npc_name: str, flow_run_id: str, inference_config: dict, inference_type: Type):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    # region ENV vars
    #grammar_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/grammar.txt'
    requests_count = int(os.getenv('STEP_2_MAX_REQUESTS_COUNT', 10))
    #use_thinking = os.getenv('STEP_2_USE_THINKING', 'false').lower() in ("1", "true", "yes", "on")
    #system_prompt_f_path = f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'

    validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation'
    # endregion

    if not os.path.isdir(validation_dataset_dir_path):
        print("No validation dataset path provided")
        exit(1)

    dataset_files = list_files(validation_dataset_dir_path)
    '''
    llm_cfg_f_path = os.getenv('STEP_2_LLM_CFG_F_PATH', '')
    llm_config = load_ullm_config(f'{llm_cfg_f_path}')
    llm_config['model'] = model
    llm_config['lora_adapter'] = lora
    llm_config['system_prompt'] = get_system_prompt(system_prompt_f_path)
    grammar_string = get_grammar(grammar_f_path)
    llm_config['grammar'] = grammar_string
    '''
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

            response_dict, think_block = ullama.chat(
                model=inference_config.get('model', ''),
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


#@task(name="step_2_lora_validation")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = os.path.abspath(f'{flow_run_dir_path}/manifest.json')

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    print(f'===> Base model inference')

    ollama_inference_cfg = {
        'model': MODEL,
        'system_prompt': get_system_prompt(f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'),
    }

    base_metrics = inference(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        inference_config=ollama_inference_cfg,
        inference_type=OllamaHelper
    )

    ullama_inference_cfg = make_ullama_config(git_commit, npc_name, flow_run_id, llm_model_f_path, lora_adapter_f_path)
    with open(os.path.join(f'{flow_run_dir_path}/', 'inference_cfg.json'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(ullama_inference_cfg, indent=2))

    print(f'===> Lora model inference')
    lora_metrics = inference(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        inference_config=ullama_inference_cfg,
        inference_type=ULlamaHelper
    )

    validation_data = {
        "validation" : {
            "base_metrics": base_metrics,
            "lora_metrics": lora_metrics
        }
    }

    update_manifest(manifest_f_path, validation_data)

    from pathlib import Path

    Path(f"{flow_run_dir_path}/reports/").mkdir(parents=True, exist_ok=True)

    # region make .md report
    md_report = generate_validation_report(
        manifest=manifest,
        metrics_base=base_metrics,
        metrics_lora=lora_metrics,
    )

    with open(os.path.join(f'{flow_run_dir_path}/reports/', 'report.md'), 'w', encoding="utf-8") as f:
        f.write(md_report)
# endregion

# region make metrics plots
    make_metrics_plot(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        metrics_model_base=base_metrics,
        metrics_model_lora=lora_metrics,
    )
# endregion

if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'

    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)
