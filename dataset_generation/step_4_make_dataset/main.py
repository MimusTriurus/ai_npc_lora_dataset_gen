import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict
import subprocess
from prefect import task

from common.constants import DATA_DIR_NAME, GEN_SYS_PROMPT_DIR_NAME, GEN_NPC_ANSWER_DIR_NAME, DATASET_DIR_NAME
from common.helpers import (
    list_files,
    load_jsonl_to_dict,
    save_dict_records_to_jsonl,
    update_manifest,
)

black_list_for_dialogs_per_action = os.getenv('STEP_4_BLACK_LIST_FOR_DIALOGS_PER_ACTION', '').split(',')

def create_dataset_record(sp, user_request: dict, npc_response: dict, use_thinking: bool = False) -> dict:
    if not use_thinking:
        del npc_response['think']
    base = {
        "messages": [
            {"role": "system", "content": sp},
            {
                "role": "user",
                "content": json.dumps(user_request)
            },
            {
                "role": "assistant",
                "content": json.dumps(npc_response)
            }
        ]
    }
    return base

@task(name="step_4_make_dataset")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    inference_sp = ''

    sp_f_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'

    with open(sp_f_path, 'r', encoding='utf-8') as f:
        inference_sp += f.read()

    dialogs_per_action_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_NPC_ANSWER_DIR_NAME}/*.jsonl'

    dialogs_by_actions_f_lst = list_files(dialogs_per_action_dir_path)

    training_dataset_size_per_action: Dict[str, int] = {}
    validation_dataset_size_per_action: Dict[str, int] = {}

    for dialog_per_action_f in dialogs_by_actions_f_lst:
        name = Path(dialog_per_action_f).stem

        is_ok = True
        for prohibited_f in black_list_for_dialogs_per_action:
            if prohibited_f and prohibited_f in dialog_per_action_f:
                print(f'==> skipping {dialog_per_action_f}')
                is_ok = False
                break
        if not is_ok:
            continue

        dataset = []

        dialogs_per_action = load_jsonl_to_dict(dialog_per_action_f)

        for dialog in dialogs_per_action:
            r = create_dataset_record(
                inference_sp,
                dialog['usr_request'],
                dialog['npc_response']
            )
            dataset.append(r)

        random.shuffle(dataset)

        n = int(len(dataset) * 0.99)
        training_dataset = dataset[:n]
        validation_dataset = dataset[n:]

        target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{DATASET_DIR_NAME}/training'
        target_fname = os.path.basename(dialog_per_action_f)

        save_dict_records_to_jsonl(
            records=list(training_dataset),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )

        training_dataset_size_per_action[name] = len(training_dataset)

        target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{DATASET_DIR_NAME}/validation'

        save_dict_records_to_jsonl(
            records=list(validation_dataset),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )

        validation_dataset_size_per_action[name] = len(validation_dataset)

    pipeline_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

    manifest = {
        'unreal_commit': git_commit,
        'npc_name': npc_name,
        'pipeline_commit': pipeline_commit[:7],
        'timestamp': datetime.now().isoformat(),
        'flow_run_id': flow_run_id,
        'dataset': {
            'training': {
                'actions': training_dataset_size_per_action,
                'total': sum(training_dataset_size_per_action.values()),
            },
            'validation': {
                'actions': validation_dataset_size_per_action,
                'total': sum(validation_dataset_size_per_action.values()),
            }
        }
    }

    manifest_f_name = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/manifest.json'

    update_manifest(manifest_f_name, manifest)

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = "trader"
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id='v1'))