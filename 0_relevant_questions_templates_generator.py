import os
import re
from typing import Set

from common.data_classes import Question
from common.helpers import read_file, save_questions_to_jsonl
from common.ollama_helper import *
import json

npc = os.getenv('NPC', 'npc_trader')
input_data = 'player_questions'
templates_f_path = os.getenv('QUESTIONS_TEMPLATES_PATH', f'resources/{npc}/0_dataset_{input_data}_configuration.json')
npc_desc_f_path = os.getenv('NPC_DESCRIPTION_PATH', f'resources/{npc}/npc_description.md')
roles_count = int(os.getenv('ROLES_COUNT', 10))

def build_system_prompt(base_system_prompt: str, character: dict) -> str:
    character_block = (
        f"Name: {character.get('name')}\n"
        f"Role ID: {character.get('id')}\n"
        f"Description: {character.get('description')}\n"
        f"Speech Style: {character.get('speech_style')}\n"
        f"Motivation: {character.get('motivation')}"
    )

    updated_prompt = base_system_prompt.replace(
        "<character_description></character_description>",
        f"<character_description>\n{character_block}\n</character_description>"
    )

    return updated_prompt

if __name__ == '__main__':
    npc_description = read_file(npc_desc_f_path)

    generation_questions = json.loads(read_file(templates_f_path))
    helper = OllamaHelper(OLLAMA_HOST)
    for qt in generation_questions['questions_templates']:
        base_template = qt['base_template']

        param_pattern = ''
        target_param = ''
        r = re.findall(r'<(.*?)>', base_template)
        if r:
            target_param = r[0]

        MAX = qt['count']
        action_name = qt['action']
        question = Question(
            base_template,
            action_name,
            qt['motivation'],
            '',
        )
        questions_templates: Set[Question] = set()
        questions_templates.add(question)

        system_prompt = read_file(f'resources/0_system_prompt_gen_{input_data}.md')
        system_prompt = system_prompt.replace('<npc_description></npc_description>', npc_description)
        print(f'= Generate analogue of: {base_template}')
        for pr in generation_questions['roles'][:roles_count]:
            pr['motivation'] = qt['motivation']
            new_system_prompt = build_system_prompt(system_prompt, pr)
            new_system_prompt += f'\n- {base_template}'

            print(f"== Role {pr['id']}")
            counter = MAX
            while counter > 0:
                template_request = f"Create another template for : {base_template}"
                prompt = build_prompt(new_system_prompt, template_request)

                new_template, think = helper.generate(MODEL, prompt)
                if new_template.startswith('"'):
                    new_template = new_template[1:]
                if new_template.endswith('"'):
                    new_template = new_template[:-1]

                is_ok = True
                if target_param:
                    r = re.findall(r'<(.*?)>', new_template)
                    for result in r:
                        if result != target_param:
                            is_ok = False
                            break

                if not is_ok:
                    print(f'===> skip (bad template ARG) {new_template}')
                    continue

                if len(new_template.split('\n')) > 1:
                    print(f'===> skip (multiline) {new_template}')
                    is_ok = False

                #new_system_prompt += f'\n* {new_template}'
                question = Question(
                    new_template,
                    action_name,
                    qt['motivation'],
                    '',
                )

                exists = any(x.template == new_template for x in questions_templates)

                if not exists:
                    print(f'=== [{MAX - counter}] {new_template}')
                    questions_templates.add(question)
                    counter -= 1
                else:
                    if new_template not in new_system_prompt:
                        new_system_prompt += f'\n- {new_template}'
                    print(f'=== WARNING: {new_template} already exists')

        save_questions_to_jsonl(
            questions_templates,
            f"resources/{npc}/output/0_request_templates/{action_name}.json"
        )
        #break
    print('=== end ===')
