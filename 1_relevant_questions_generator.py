from common.data_classes import Question, UserRequest, Action, NpcResponse, RequestResponsePair
from common.helpers import read_file, extract_angle_bracket_substrings, save_dataclass_records_to_jsonl
from common.ollama_helper import *
import json

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

if __name__ == '__main__':
    system_prompt_template = read_file('resources/systemPrompt.md')
    user_desc = read_file('resources/user_description.md')
    npc_desc = read_file('resources/npc_trader/npc_description.md')
    chat_example = read_file('resources/npc_trader/chat_example.md')
    actions_text = read_file('resources/npc_trader/actions.txt')

    system_prompt = system_prompt_template.replace('<npc_description></npc_description>', npc_desc)
    system_prompt = system_prompt.replace('<user_description></user_description>', user_desc)
    system_prompt = system_prompt.replace('<chat_example></chat_example>', chat_example)
    system_prompt = system_prompt.replace('<actions></actions>', actions_text)

    questions_templates = load_questions_from_jsonl(
        'resources/npc_trader/output/0_generated_player_questions_templates.json')

    context = json.loads(read_file('resources/npc_trader/dataset_context.json'))

    context_actions = context['actions']
    context_params = context['params']

    rrps = list()

    for question_template in questions_templates:
        if not question_template.template:
            continue
        if question_template.action in context_actions:
            question_template_str = question_template.template
            user_motivation = question_template.motivation
            extracted_args = extract_angle_bracket_substrings(question_template_str)
            if len(extracted_args) > 1 or len(extracted_args) == 0:
                continue
            extracted_arg = extracted_args[0]
            if extracted_arg not in context_params:
                continue

            extracted_params = context_params[extracted_arg]

            action_data = context_actions[question_template.action]
            actions_contexts = action_data['actions_contexts']

            for parameter in extracted_params:
                user_request = question_template.template.replace(extracted_arg, parameter)
                for action_context in actions_contexts:
                    context = action_context['context']
                    request_context = context.replace(extracted_arg, parameter)

                    npc_callback_action = Action.parse_action(action_context['action'])
                    for i in range(len(npc_callback_action.parameters)):
                        if npc_callback_action.parameters[i] == extracted_arg:
                            npc_callback_action.parameters[i] = parameter
                    request = UserRequest(
                        context=user_motivation,
                        state_of_user=request_context,
                        request_of_user=user_request,
                    )
                    response = NpcResponse(
                        emotion='',
                        answer='',
                        action=npc_callback_action,
                    )
                    rrp = RequestResponsePair(user_request=request, npc_response=response)
                    rrps.append(rrp)
        else:
            npc_callback_action = Action.parse_action(question_template.action)
            request = UserRequest(
                context=question_template.motivation,
                state_of_user=question_template.motivation,
                request_of_user=question_template.template,
            )
            response = NpcResponse(
                emotion='',
                answer='',
                action=npc_callback_action,
            )
            rrp = RequestResponsePair(user_request=request, npc_response=response)
            rrps.append(rrp)

    save_dataclass_records_to_jsonl(rrps, output_file='resources/npc_trader/output/1_generated_relevant_player_requests.json')

    print('end of generating relevant questions')