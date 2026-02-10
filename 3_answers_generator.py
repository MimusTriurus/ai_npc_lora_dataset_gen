import random
import re

from common.data_classes import (
    Action,
    NpcResponse,
    RequestResponsePair
)
from common.helpers import (
    read_file,
    save_dataclass_records_to_jsonl,
    load_jsonl_to_dataclasses, list_files
)
from common.ollama_helper import *
import json

npc = os.getenv('NPC', 'npc_trader')
relevant_limit = int(os.getenv('RELEVANT_LIMIT', 99999))
irrelevant_limit = int(os.getenv('IRRELEVANT_LIMIT', 99999))
inference_sys_prompt = os.getenv('SYS_PROMPT_F_PATH', 'resources/npc_trader/system_prompt.md')

def generate_answers(
        sp_f_path: str,
        ce_f_path: str,
        rr_f_path: str,
        limit: int = 99999
) -> List[RequestResponsePair]:
    system_prompt_template = read_file(sp_f_path)
    user_desc = read_file('resources/user_description.md')
    npc_desc = read_file(f'resources/{npc}/npc_description.md')
    chat_example = read_file(ce_f_path)
    actions = read_file(f'resources/{npc}/actions.txt')

    system_prompt = system_prompt_template.replace('<npc_description></npc_description>', npc_desc)
    system_prompt = system_prompt.replace('<user_description></user_description>', user_desc)
    system_prompt = system_prompt.replace('<chat_example></chat_example>', chat_example)
    system_prompt = system_prompt.replace('<actions></actions>', actions)

    helper = OllamaHelper(OLLAMA_HOST)

    rrps = load_jsonl_to_dataclasses(
        rr_f_path,
        RequestResponsePair
    )
    random.shuffle(rrps)
    min_size = min(len(rrps), limit)
    rrps = rrps[:min_size]

    for i in (range(len(rrps))):
        valid_action_dict = rrps[i].npc_response['action']
        valid_action = Action(
                    name=valid_action_dict['name'],
                    parameters=valid_action_dict['parameters'],
                )

        json_request = json.dumps(rrps[i].user_request)
        new_system_prompt = build_prompt(system_prompt, json_request)
        response, think = helper.generate(MODEL, new_system_prompt)

        if think is not None:
            think = think.replace('\n', ' ')
            think = re.sub(r"[^\x00-\x7F]", " ", think)
            think = think.rstrip().lstrip().lstrip()
        else:
            think = ''

        response = re.sub(r"[^\x00-\x7F]", " ", response)
        response = response.replace('</think>', '').replace('<think>', '')
        response = response.rstrip().lstrip()

        rrps[i].user_request['request_of_user'] = re.sub(r"[^\x00-\x7F]", " ", rrps[i].user_request['request_of_user'])

        action: Action = None
        answer: str = None
        emotion: str = None

        try:
            obj = json.loads(response)
            if 'emotion' in obj and 'answer' in obj:
                answer = f"{obj['answer']}"
                emotion = obj['emotion']
                action_dict = rrps[i].npc_response['action']
                action = Action(
                    name=action_dict['name'],
                    parameters=action_dict['parameters'],
                )

                if action != valid_action:
                    raise Exception(f"Invalid action: {action} != {valid_action}!!!")

        except Exception as e:
            print(f'--> Skip. Error: {e}\n request: {json_request}\n response:\n{response}')
            action = None
            answer = None
            emotion = None
            continue
        #break

        if action is not None and answer is not None and emotion is not None:
            print(f'=== {action.name} [{i}/{len(rrps) - 1}] ===')
            rrps[i].npc_response = NpcResponse(
                answer=answer,
                emotion=emotion,
                action=action,
                think=think,
            )

    return rrps

if __name__ == '__main__':
    generate_relevant_answers = os.getenv('GENERATE_IRRELEVANT_ANSWERS', 'true').lower() in ("1", "true", "yes", "on")
    generate_irrelevant_answers = os.getenv('GENERATE_IRRELEVANT_ANSWERS', 'false').lower() in ("1", "true", "yes", "on")

    if generate_relevant_answers:
        print(f'=== Generate relevant answers ===')
        relevant_requests_f_paths = list_files(f'resources/{npc}/output/1_requests')
        for f_path in relevant_requests_f_paths:
            print(f'=== {f_path} ===')
            relevant_cases = generate_answers(
                inference_sys_prompt,
                f'resources/{npc}/chat_example.md',
                f_path,
                relevant_limit
            )
            filename = os.path.basename(f_path)
            save_dataclass_records_to_jsonl(relevant_cases, output_file=f'resources/{npc}/output/3_requests_responses/{filename}')

    print('\n')

    if generate_irrelevant_answers:
        print(f'=== Generate irrelevant answers ===')
        irrelevant_requests_f_paths = list_files(f'resources/{npc}/output/2_irrelevant_requests')
        for f_path in irrelevant_requests_f_paths:
            print(f'=== {f_path} ===')
            irrelevant_cases = generate_answers(
                'resources/2_systemPrompt_irrelevant_case.md',
                'resources/2_chat_example_irrelevant_case.md',
                f_path,
                irrelevant_limit
            )
            filename = os.path.basename(f_path)
            save_dataclass_records_to_jsonl(irrelevant_cases, output_file=f'resources/{npc}/output/3_requests_responses/{filename}')

    print('=== end ===')