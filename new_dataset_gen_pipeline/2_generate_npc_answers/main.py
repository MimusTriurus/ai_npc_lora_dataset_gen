import json
import os
from dataclasses import asdict

from common.data_structures import *
from common.helpers import list_files, load_jsonl_to_dataclasses, extract_nsloctext_value, save_dict_records_to_jsonl, \
    replace_unicode
from common.ollama_helper import MODEL, OLLAMA_HOST, OllamaHelper
from common.template_gen_components import env

usr_requests_dir_path = os.getenv('USR_REQUESTS_DIR_PATH', '')
black_list_for_usr_request = os.getenv('BLACK_LIST_FOR_USR_REQUESTS', '').split(',')
answer_gen_sp_template_f_path = os.getenv('ANSWER_GEN_SP_TEMPLATE_F_PATH', '')
actions_f_path = os.getenv('ACTIONS_F_PATH', '')
actions_desc_f_path = os.getenv('ACTIONS_DESC_F_PATH', '')
npc_name = os.getenv('NPC_NAME', '')

if __name__ == '__main__':
    answer_gen_sp_template = ''
    with open(answer_gen_sp_template_f_path, 'r', encoding='utf-8') as f:
        answer_gen_sp_template += f.read()

    npc_description = ''

    with open(actions_f_path) as f:
        for npc_data in json.load(f):
            if npc_data['Name'] == npc_name:
                npc_description = extract_nsloctext_value(npc_data['Description'])

    actions_desc: Dict[str, str] = {}
    with open(actions_desc_f_path) as f:
        actions_desc.update(json.loads(f.read()))

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

        player_requests = load_jsonl_to_dataclasses(usr_request_f, Root)

        pr_template = env.from_string(answer_gen_sp_template)
        for pr in player_requests:
            action_obj = Action(
                name=pr.npc_valid_action['action']['name'],
                parameters=pr.npc_valid_action['action']['parameters'],
            )

            player_role_obj = PlayerRole(
                name=pr.player_role['name'],
                description=pr.player_role['description'],
                speech_style=pr.player_role['speech_style'],
            )

            action_name: str = action_obj.name
            player_role_str: str = str(player_role_obj)
            player_intention_str: str = actions_desc.get(action_name, '')
            npc_action_str = str(action_obj)

            template_params = {
                'npc_description': npc_description,
                'player_role': player_role_str,
                'player_request': json.dumps(pr.usr_request, indent=2),
                'player_intention': player_intention_str,
                'npc_action_str': npc_action_str,
            }

            inference_system_prompt = pr_template.render(template_params)

            helper = OllamaHelper(OLLAMA_HOST)
            response_str, think = helper.generate(MODEL, inference_system_prompt)
            response = json.loads(response_str)

            pr.npc_valid_action['emotion'] = response['emotion']
            pr.npc_valid_action['answer'] = replace_unicode(response['answer'])

            dataset_record = asdict(pr)

            out_requests.append(dataset_record)

            print(f'USR: {pr.usr_request["request"]}')
            print(f'--- ACTION: {npc_action_str} ---')
            print(f'NPC: {response["answer"]}')
            print()

        target_dir = f'output_data/{npc_name}/2_generate_npc_answers'
        target_fname = os.path.basename(usr_request_f)

        save_dict_records_to_jsonl(
            records=list(out_requests),
            output_file=target_fname,
            folder_path=target_dir,
            append=True
        )
