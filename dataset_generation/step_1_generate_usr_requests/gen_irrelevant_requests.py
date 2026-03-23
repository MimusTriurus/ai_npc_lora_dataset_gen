import json
import math
from typing import Set

from prefect import task

from common.constants import DATA_DIR_NAME, GEN_USR_REQUEST_DIR_NAME, ACTION_FOR_IRRELEVANT_REQUESTS
from common.helpers import get_npc_data, save_dict_records_to_jsonl, replace_unicode
from common.ollama_helper import *
from common.template_gen_components import env

def get_roles() -> List[dict]:
    usr_roles_f_path = os.getenv('STEP_1_USR_ROLES_F_PATH', 'resources/user_roles.json')
    with open(f"{usr_roles_f_path}", "r", encoding="utf-8") as f:
        roles: List[dict] = json.load(f)
        return roles

def get_system_prompt_template() -> str:
    sp_template_f_path = os.getenv(
        'STEP_1_NEGATIVE_SP_TEMPLATE_F_PATH',
        'dataset_generation/step_1_generate_usr_requests/gen_usr_negative_requests_system_prompt.j2'
    )
    with open(sp_template_f_path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(
    player_role: Dict,
    npc_role: str,
    action: str,
    num_requests: int
) -> str:
    system_prompt_template = get_system_prompt_template()
    sp_template = env.from_string(system_prompt_template)
    params = {
        "action": action,
        "player_role_json": json.dumps(player_role, indent=2, ensure_ascii=False),
        "npc_role_json": npc_role,
        "num_requests": num_requests
    }
    rendered_template = sp_template.render(params)
    return rendered_template

@task(name="step_1_generate_irrelevant_usr_requests")
def process(git_commit: str, npc_name: str, flow_run_id: str, dataset_size_per_action: int):
    roles = get_roles()
    roles_count = len(roles)

    requests_per_role = math.ceil(dataset_size_per_action / roles_count)
    requests_per_role = max(25, requests_per_role)

    cfg = {}
    helper = OllamaHelper(cfg)

    questions_per_role = list()

    for r in roles:
        npc_data = get_npc_data(git_commit, npc_name, flow_run_id)
        npc_desc = npc_data['Description']

        prompt = build_system_prompt(
            player_role=r,
            npc_role=npc_desc,
            action=ACTION_FOR_IRRELEVANT_REQUESTS,
            num_requests=requests_per_role
        )

        questions: Set[str] = set()

        while len(questions) < requests_per_role:
            questions_set = set()
            try:
                questions_str, think = helper.generate(MODEL, prompt)
                questions_set = set(json.loads(questions_str))
            except Exception as e:
                print(e)
            questions.update(questions_set)

        for q in questions:
            usr_request = {
                "request": replace_unicode(q),
                "usr_state": '',
                "npc_state": '',
            }

            npc_valid_action = {
                "emotion": "",
                "answer": "",
                "think": "",
                "action": {
                    "name": ACTION_FOR_IRRELEVANT_REQUESTS,
                    "parameters": {}
                }
            }

            request_per_action = {
                'usr_request': usr_request,
                'npc_response': npc_valid_action,
                'player_role': r
            }

            questions_per_role.append(request_per_action)

        if len(questions_per_role) >= dataset_size_per_action:
            break

    if len(questions_per_role) < dataset_size_per_action:
        print(f'Warning!!! irrelevant database is less than necessary: {len(questions_per_role)} < {dataset_size_per_action}')

    target_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_USR_REQUEST_DIR_NAME}'
    target_fname = f'{ACTION_FOR_IRRELEVANT_REQUESTS}.jsonl'

    save_dict_records_to_jsonl(
        records=questions_per_role[:dataset_size_per_action],
        output_file=target_fname,
        folder_path=target_dir,
        append=True
    )

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = "trader"
    FLOW_RUN_ID = "v_test"
    DATASET_SIZE_PER_ACTION = 115
    process(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id=FLOW_RUN_ID,
        dataset_size_per_action=DATASET_SIZE_PER_ACTION,
    )
    exit()