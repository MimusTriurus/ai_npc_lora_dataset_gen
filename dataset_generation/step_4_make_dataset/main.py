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

    VAL_RATIO = 0.1

    # Phase 1: collect records by composite key (action name + parameters)
    # records_by_composite: "SellItem|{"item":"pistol"}" -> [(file_stem, record), ...]
    # variants_by_action:   "SellItem" -> ["SellItem|{...}", ...]
    records_by_composite: Dict[str, list] = {}
    variants_by_action: Dict[str, list] = {}

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

        dialogs_per_action = load_jsonl_to_dict(dialog_per_action_f)

        for dialog in dialogs_per_action:
            r = create_dataset_record(
                inference_sp,
                dialog['usr_request'],
                dialog['npc_response']
            )
            action = dialog['npc_response'].get('action') or {}
            action_name = action.get('name', 'unknown')
            composite_key = action_name + '|' + json.dumps(action.get('parameters', {}), sort_keys=True)

            if composite_key not in records_by_composite:
                records_by_composite[composite_key] = []
                variants_by_action.setdefault(action_name, []).append(composite_key)
            records_by_composite[composite_key].append((name, r))

    # Phase 2: stratified split — equal per variant within each action, min 1 per variant
    # n_val_per_variant = max(1, int(min_variant_size * VAL_RATIO))
    training_by_file: Dict[str, list] = {}
    validation_by_action: Dict[str, list] = {}
    validation_dataset_size_per_action: Dict[str, int] = {}

    for action_name, composite_keys in variants_by_action.items():
        min_variant_size = min(len(records_by_composite[k]) for k in composite_keys)
        n_val_per_variant = max(1, int(min_variant_size * VAL_RATIO))

        validation_by_action[action_name] = []

        for composite_key in composite_keys:
            items = records_by_composite[composite_key]
            random.shuffle(items)

            for _, record in items[:n_val_per_variant]:
                validation_by_action[action_name].append(record)

            for file_name, record in items[n_val_per_variant:]:
                training_by_file.setdefault(file_name, []).append(record)

            params = composite_key.split('|', 1)[1]
            print(f'  val {action_name} [{params}]: {n_val_per_variant}')

        total = len(validation_by_action[action_name])
        validation_dataset_size_per_action[action_name] = total
        print(f'val {action_name} total: {total}')

    # Phase 3: save training per source file, validation per action name
    training_dataset_size_per_action: Dict[str, int] = {}

    for file_name, records in training_by_file.items():
        target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{DATASET_DIR_NAME}/training'
        save_dict_records_to_jsonl(
            records=records,
            output_file=f'{file_name}.jsonl',
            folder_path=target_dir,
            append=True
        )
        training_dataset_size_per_action[file_name] = len(records)

    target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{DATASET_DIR_NAME}/validation'
    for action_name, records in validation_by_action.items():
        random.shuffle(records)
        save_dict_records_to_jsonl(
            records=records,
            output_file=f'{action_name}.jsonl',
            folder_path=target_dir,
            append=True
        )

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
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID))