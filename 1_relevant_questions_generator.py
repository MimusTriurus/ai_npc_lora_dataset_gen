from unidecode import unidecode
from common.data_classes import Question, UserRequest, Action, NpcResponse, RequestResponsePair
from common.helpers import read_file, extract_angle_bracket_substrings, save_dataclass_records_to_jsonl, list_files
from common.ollama_helper import *
import json
import re
import random
import os

npc = os.getenv('NPC', 'npc_trader')

def fill_ranges(text: str) -> str:
    if "[" not in text or "]" not in text:
        return text

    def replace(match):
        low, high = map(int, match.group(1).split('-'))
        return str(random.randint(low, high))

    return re.sub(r"\[(\d+-\d+)\]", replace, text)

def fill_template(template: str, inputs: list[str]) -> str:
    match = re.search(r"\{(.*)\}", template)
    if not match:
        return template

    block = match.group(1)

    items = []
    for value in inputs:
        filled = re.sub(r"<[^>]+>", value, block)
        filled = fill_ranges(filled)
        items.append(f'\n - {filled}')

    filled_block = "".join(items)

    result = re.sub(r"\{.*\}", filled_block, template)

    return result

def load_questions_from_jsonl(path: str) -> List[Question]:
    questions: list[Question] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            try:
                questions.append(
                    Question(
                        template=obj["base_template"],
                        action=obj["action"],
                        motivation=obj["motivation"],
                        context=obj["context"] if "context" in obj else "",
                    )
                )
            except KeyError as e:
                raise ValueError(
                    f"Missing field {e} on line {line_no}"
                ) from e

    return questions

def load_questions_from_directory(path: str) -> list[Question]:
    questions: list[Question] = []
    file_paths = list_files(path)
    for file_path in file_paths:
        part = load_questions_from_jsonl(file_path)
        questions.extend(part)
    return questions

def make_context_from_template(context_template: str, context_params: dict) -> str:
    arg = extract_angle_bracket_substrings(context_template)
    if arg:
        params  = context_params.get(arg[0])
        context = fill_template(context_template, params)
        return context
    return context_template

if __name__ == '__main__':
    questions_templates = load_questions_from_directory(
        f'resources/{npc}/output/0_request_templates'
    )

    context_templates = json.loads(read_file(f'resources/{npc}/1_dataset_context.json'))

    context_actions = context_templates['actions']
    context_params: dict = context_templates['params']

    for question_template in questions_templates:
        if not question_template.template or not question_template.action or not question_template.motivation:
            continue

        action_data = context_actions.get(question_template.action)

        if action_data:
            question_template_str = question_template.template
            user_motivation = question_template.motivation

            actions_contexts = action_data['actions_contexts']

            for action_context_template in actions_contexts:
                action_context = make_context_from_template(
                    action_context_template['context'],
                    context_params
                )

                rrps = list()

                action_template = Action.parse_action(action_context_template['action'])
                action_template_arg = action_template.parameters[0] if len(action_template.parameters) > 0 else None
                if not action_template_arg:
                    print(f"==> Error. Action {action_context['action']} has no parameters!!!")
                    continue
                args: List[str] = context_params.get(action_template_arg, [])
                if not args:
                    args = [action_template_arg]

                for arg in args:
                    user_request = question_template.template.replace(action_template_arg, arg)

                    state_of_user = action_context_template['state_of_user']

                    context_template = fill_ranges(action_context)
                    state_of_user = fill_ranges(state_of_user)

                    state_of_user = state_of_user.replace(action_template_arg, arg)
                    context_templates = context_template.replace(action_template_arg, arg)

                    action_template = Action.parse_action(action_context_template['action'])
                    npc_callback_action = action_template
                    for i in range(len(npc_callback_action.parameters)):
                        if npc_callback_action.parameters[i] == action_template_arg:
                            npc_callback_action.parameters[i] = arg
                    request = UserRequest(
                        context=context_templates,
                        state_of_user=state_of_user,
                        request_of_user=unidecode(user_request),
                    )
                    response = NpcResponse(
                        emotion='',
                        answer='',
                        action=npc_callback_action,
                        think='',
                    )
                    rrp = RequestResponsePair(user_request=request, npc_response=response)
                    rrps.append(rrp)

                    save_dataclass_records_to_jsonl(
                        rrps,
                        output_file=f'resources/{npc}/output/1_requests/{npc_callback_action.name}.json',
                        append=True
                    )

    print('end of generating relevant questions')