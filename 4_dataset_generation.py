import os
import random

from common.helpers import (
    load_jsonl_to_dataclasses,
    read_file,
    save_dict_records_to_jsonl, list_files
)
from common.data_classes import *
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL

import json
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl_with_escaped_json(path: str) -> List[Dict[str, Any]]:
    result = []
    p = Path(path)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            outer = json.loads(line)

            inner = json.loads(outer)

            result.append(inner)

    return result


def summarize_thinking(input_thinking: str) -> str:
    prompt = f'''
Role: Dataset Optimization Expert.
Task: Rewrite the <think> block to be extremely blunt, compact, and factual.
Rules for the new <think> block:
Max 2 short sentences.
No phrases like "The user is asking...", "I should...", "Okay, let's break this down".
Format: [Fact/Input] -> [Logic/Check] -> [Result/Action].
Use telegraphic style.
Input Example: "The user is asking if the trader has an adrenaline shot. The state_of_user says the user has 100 gold. The context says the trader doesn't have it and it's sold out. The action is SoldOut."
Desired Output: "User requests adrenaline_shot. User has 100 gold and trader has 1 Item sold out in context. Action: SoldOut. Emotion: Neutral."
Your specific task: Rewrite the following block: {input_thinking}
    '''
    helper = OllamaHelper(OLLAMA_HOST)
    response, think = helper.generate(MODEL, prompt)
    return response


def create_dataset_record(sp, context, state, request, think_content, emotion, answer, action, use_thinking=True):
    user_input_json = {
        "context": context,
        "state_of_user": state,
        "request_of_user": request
    }
    assistant_output_json = {
        "emotion": emotion,
        "answer": answer,
        "action": action
    }
    if use_thinking:
        think_content = summarize_thinking(think_content)
        assistant_content = f"<think>\n{think_content}\n</think>\n{json.dumps(assistant_output_json, ensure_ascii=False)}"
    else:
        assistant_content = f"{json.dumps(assistant_output_json, ensure_ascii=False)}"

    base = {
        "messages": [
            {"role": "system", "content": sp},
            {
                "role": "user",
                "content": json.dumps(user_input_json, ensure_ascii=False)
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    return base

if __name__ == '__main__':
    npc = os.getenv('NPC', 'npc_trader')
    use_thinking = os.getenv("USE_THINKING", "false").lower() in ("1", "true", "yes", "on")

    system_prompt_template = read_file('resources/systemPrompt_for_LoRA_training.md')
    npc_desc = read_file(f'resources/{npc}/npc_description.md')
    actions_text = read_file(f'resources/{npc}/actions.txt')

    system_prompt = system_prompt_template.replace('<npc_description></npc_description>', npc_desc)
    system_prompt = system_prompt.replace('<actions></actions>', actions_text)

    think_instruction = '6. Think concisely in <think> tags (Facts -> Logic -> Action).' if use_thinking else ''

    system_prompt = system_prompt.replace('<think_instruction></think_instruction>', think_instruction)

    relevant_requests_f_paths = list_files(f'resources/{npc}/output/3_requests_responses')
    for f_path in relevant_requests_f_paths:
        print(f'=== {f_path} ===')
        rrs = load_jsonl_to_dataclasses(
            f_path,
            RequestResponsePair
        )

        random.shuffle(rrs)

        dataset = []

        rr_by_action = dict()
        counter = 1

        path = Path(f_path)
        dataset_name = path.stem

        for rr in rrs:
            print(f'{counter} / {len(rrs)}')
            counter += 1
            if not rr.npc_response['answer']:
                continue
            ds_json = create_dataset_record(
                sp=system_prompt,
                context=rr.user_request['context'],
                state=rr.user_request['state_of_user'],
                request=rr.user_request['request_of_user'],
                think_content=rr.npc_response['think'],
                emotion=rr.npc_response['emotion'],
                answer=rr.npc_response['answer'],
                action=rr.npc_response['action'],
                use_thinking=use_thinking
            )

            dataset.append(ds_json)

        n = int(len(dataset) * 0.95)
        training_dataset = dataset[:n]
        validation_dataset = dataset[n:]

        dir_name = 'instruct_dataset' if not use_thinking else 'thinking_dataset'

        save_dict_records_to_jsonl(training_dataset, f'resources/{npc}/output/{dir_name}/training/{dataset_name}.jsonl')
        save_dict_records_to_jsonl(validation_dataset, f'resources/{npc}/output/{dir_name}/validation/{dataset_name}.jsonl')

    print('=== end ===')