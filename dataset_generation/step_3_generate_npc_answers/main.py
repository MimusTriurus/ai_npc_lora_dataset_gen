import json
import os
from typing import Dict
from prefect import task

from common.constants import DATA_DIR_NAME, GEN_USR_REQUEST_DIR_NAME, GEN_SYS_PROMPT_DIR_NAME, GEN_NPC_ANSWER_DIR_NAME
from common.data_classes import Action, PlayerRole
from common.helpers import (
    list_files,
    extract_nsloctext_value,
    save_dict_records_to_jsonl,
    replace_unicode,
    load_jsonl_to_dict
)
from common.ollama_helper import OLLAMA_HOST, OllamaHelper
from common.template_gen_components import env

black_list_for_usr_request = os.getenv('STEP_3_BLACK_LIST_FOR_USR_REQUESTS', '').split(',')
answer_gen_sp_template_f_path = os.getenv('STEP_3_ANSWER_GEN_SP_TEMPLATE_F_PATH', '')
MODEL = os.getenv('STEP_3_OLLAMA_MODEL')

@task(name="step_3_generate_npc_answers")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    answer_gen_sp_template = ''
    with open(answer_gen_sp_template_f_path, 'r', encoding='utf-8') as f:
        answer_gen_sp_template += f.read()

    npc_description = ''

    actions_f_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/description.json'

    with open(actions_f_path) as f:
        npc_data = json.load(f)
        if npc_data['Name'] == npc_name:
            npc_description = extract_nsloctext_value(npc_data['Description'])

    actions_desc: Dict[str, str] = {}

    actions_desc_f_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_SYS_PROMPT_DIR_NAME}/actions_desc.json'

    with open(actions_desc_f_path) as f:
        actions_desc.update(json.loads(f.read()))

    usr_requests_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_USR_REQUEST_DIR_NAME}/*.jsonl'

    usr_requests_by_actions_f_lst = list_files(usr_requests_dir_path)
    for usr_request_f in usr_requests_by_actions_f_lst:
        is_ok = True
        for prohibited_f in black_list_for_usr_request:
            if prohibited_f and prohibited_f in usr_request_f:
                print(f'==> skipping {usr_request_f}')
                is_ok = False
                break
        if not is_ok:
            continue

        out_requests = []

        player_requests = load_jsonl_to_dict(usr_request_f)

        pr_template = env.from_string(answer_gen_sp_template)
        for pr in player_requests:
            action_obj = Action(
                name=pr['npc_response']['action']['name'],
                parameters=pr['npc_response']['action']['parameters'],
            )

            player_role_obj = PlayerRole(
                name=pr['player_role']['name'],
                description=pr['player_role']['description'],
                speech_style=pr['player_role']['speech_style'],
            )

            action_name: str = action_obj.name
            player_role_str: str = str(player_role_obj)
            player_intention_str: str = actions_desc.get(action_name, '')
            npc_action_str = str(action_obj)

            template_params = {
                'npc_description': npc_description,
                'player_role': player_role_str,
                'player_request': pr['usr_request'],
                'player_intention': player_intention_str,
                'npc_action_str': npc_action_str,
            }

            inference_system_prompt = pr_template.render(template_params)

            helper = OllamaHelper(OLLAMA_HOST)
            response_str, think = helper.generate(MODEL, inference_system_prompt)
            response = json.loads(response_str)

            pr['npc_response']['emotion'] = response['emotion']
            pr['npc_response']['answer'] = replace_unicode(response['answer'])

            out_requests.append(pr)
            '''
            print(f'USR: {pr["usr_request"]["request"]}')
            print(f'--- ACTION: {npc_action_str} ---')
            print(f'NPC: {response["answer"]}')
            print()
            '''

        target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_NPC_ANSWER_DIR_NAME}'
        target_fname = os.path.basename(usr_request_f)

        save_dict_records_to_jsonl(
            records=list(out_requests),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = "trader"
    FLOW_RUN_ID = "v_test"
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID))
