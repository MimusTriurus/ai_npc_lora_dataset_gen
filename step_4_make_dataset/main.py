import json
import os
import random

from prefect import task
from common.helpers import (
    list_files,
    load_jsonl_to_dataclasses,
    load_jsonl_to_dict,
    save_dict_records_to_jsonl,
)

black_list_for_dialogs_per_action = os.getenv('BLACK_LIST_FOR_DIALOGS_PER_ACTION', '').split(',')
npc_name = os.getenv('NPC_NAME', '')

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

@task
def process(git_commit: str, npc_name: str):
    inference_sp = ''

    sp_f_path = f'input_data/{git_commit}/{npc_name}/1_generate_system_prompt_data/system_prompt.txt'

    with open(sp_f_path, 'r', encoding='utf-8') as f:
        inference_sp += f.read()

    dialogs_per_action_dir_path = f'input_data/{git_commit}/{npc_name}/2_generate_npc_answers/*.jsonl'

    dialogs_by_actions_f_lst = list_files(dialogs_per_action_dir_path)
    for dialog_per_action_f in dialogs_by_actions_f_lst:
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

        target_dir = f'input_data/{git_commit}/{npc_name}/3_make_dataset/training'
        target_fname = os.path.basename(dialog_per_action_f)

        save_dict_records_to_jsonl(
            records=list(training_dataset),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )

        target_dir = f'input_data/{git_commit}/{npc_name}/3_make_dataset/validation'

        save_dict_records_to_jsonl(
            records=list(validation_dataset),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_NAME = "trader"
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME))