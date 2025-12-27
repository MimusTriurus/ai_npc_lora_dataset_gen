from common.data_classes import (
    Action,
    NpcResponse,
    RequestResponsePair
)
from common.helpers import (
    read_file,
    save_dataclass_records_to_jsonl,
    load_jsonl_to_dataclasses
)
from common.ollama_helper import *
import json

def generate_answers(
        sp_f_path: str,
        ce_f_path: str,
        rr_f_path: str,
) -> List[RequestResponsePair]:
    system_prompt_template = read_file(sp_f_path)
    user_desc = read_file('resources/user_description.md')
    npc_desc = read_file('resources/npc_trader/npc_description.md')
    chat_example = read_file(ce_f_path)

    system_prompt = system_prompt_template.replace('<npc_description></npc_description>', npc_desc)
    system_prompt = system_prompt.replace('<user_description></user_description>', user_desc)
    system_prompt = system_prompt.replace('<chat_example></chat_example>', chat_example)

    helper = OllamaHelper(OLLAMA_HOST)

    rrps = load_jsonl_to_dataclasses(
        rr_f_path,
        RequestResponsePair
    )

    print(f'=== Generate answers ===')
    for i in (range(len(rrps))):
        json_request = json.dumps(rrps[i].user_request)
        new_system_prompt = build_prompt(system_prompt, json_request)
        response, think = helper.generate(MODEL, new_system_prompt)
        #print(f'Request: {json_request}')
        #print(f'Response: {response}')

        action: Action = None
        answer: str = None
        emotion: str = None

        try:
            obj = json.loads(response)
            if 'emotion' in obj and 'answer' in obj:
                answer = obj['answer']
                emotion = obj['emotion']
                action_dict = rrps[i].npc_response['action']
                action = Action(
                    name=action_dict['name'],
                    parameters=action_dict['parameters'],
                )
        except Exception as e:
            print(f'--> Skip. Error: {e}\n request: {json_request}\n response:\n{response}')
            action = None
            answer = None
            emotion = None
            continue
        #break

        if action is not None and answer is not None and emotion is not None:
            print(f'=== [{i}/{len(rrps) - 1}] ===')
            rrps[i].npc_response = NpcResponse(
                answer=answer,
                emotion=emotion,
                action=action
            )

    return rrps

if __name__ == '__main__':
    irrelevant_case = generate_answers(
        'resources/systemPrompt_irrelevant_case.md',
        'resources/chat_example_irrelevant_case.md',
        'resources/npc_trader/output/2_generated_irrelevant_player_requests.json'
    )

    relevant_case = generate_answers(
        'resources/systemPrompt.md',
        'resources/npc_trader/chat_example.md',
        'resources/npc_trader/output/1_generated_relevant_player_requests.json'
    )

    save_dataclass_records_to_jsonl(relevant_case, output_file='resources/npc_trader/output/3_generated_relevant_requests_responses.json')
    save_dataclass_records_to_jsonl(irrelevant_case, output_file='resources/npc_trader/output/3_generated_irrelevant_requests_responses.json')

    print('=== end ===')