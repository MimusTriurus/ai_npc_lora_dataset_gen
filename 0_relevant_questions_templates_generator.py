from typing import Set

from common.data_classes import Question
from common.helpers import read_file, save_questions_to_jsonl
from common.ollama_helper import *
import json

def build_system_prompt(base_system_prompt: str, character: dict) -> str:
    # Build formatted character description block
    character_block = (
        f"Name: {character.get('name')}\n"
        f"Role ID: {character.get('id')}\n"
        f"Description: {character.get('description')}\n"
        f"Speech Style: {character.get('speech_style')}\n"
        f"Motivation: {character.get('motivation')}"
    )

    # Insert into system prompt
    updated_prompt = base_system_prompt.replace(
        "<character_description></character_description>",
        f"<character_description>\n{character_block}\n</character_description>"
    )

    return updated_prompt

input_data = 'player_questions'
#input_data = 'npc_answers'


if __name__ == '__main__':
    generation_questions = json.loads(read_file(f"resources/npc_trader/dataset_{input_data}_configuration.json"))
    helper = OllamaHelper(OLLAMA_HOST)
    questions_templates: Set[Question] = set()
    for qt in generation_questions['questions_templates']:
        base_template = qt['base_template']
        MAX = qt['count']

        question = Question(
            base_template,
            qt['action'],
            qt['motivation'],
            '',
        )
        questions_templates.add(question)

        system_prompt = read_file(f'resources/system_prompt_gen_{input_data}.md')
        system_prompt = system_prompt.replace('<npc_description></npc_description>', generation_questions['npc_desc'])
        print(f'= Generate analogue of: {base_template}')
        for pr in generation_questions['roles']:
            pr['motivation'] = qt['motivation']
            new_system_prompt = build_system_prompt(system_prompt, pr)

            template_request = f"Create another template for : {base_template}"
            new_system_prompt += f'\n* {base_template}'
            print(f"== Role {pr}")
            prompt = build_prompt(new_system_prompt, template_request)
            counter = MAX
            while counter:
                new_template, think = helper.generate(MODEL, prompt)
                if new_template.startswith('"'):
                    new_template = new_template[1:]
                if new_template.endswith('"'):
                    new_template = new_template[:-1]

                if new_template not in questions_templates:
                    print(f'=== [{MAX - counter}] {new_template}')
                    new_system_prompt += f'\n* {new_template}'
                    counter -= 1
                    question = Question(
                        new_template,
                        qt['action'],
                        qt['motivation'],
                        '',
                    )
                    questions_templates.add(question)
            #break
        #break

    save_questions_to_jsonl(questions_templates, f"resources/npc_trader/output/0_generated_{input_data}_templates.json")
    print('end')
