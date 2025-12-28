import re
from typing import Set

from common.data_classes import UserRequest, Action, RequestResponsePair, NpcResponse
from common.helpers import read_file, save_dataclass_records_to_jsonl
from common.ollama_helper import *

npc = os.getenv('NPC', 'npc_trader')

def make_negative_prompt(npc_description: str, records_count: int) -> str:
    prompt = f'''
NPC: {npc_description}.

Generate exactly {records_count} deliberately irrelevant player question (something the NPC cannot possibly know, such as modern technology, the internet, or meta‑questions about being an NPC).

Rules:
- The question must be clearly outside the NPC’s knowledge domain.
- The questions should not be related in meaning.
- Questions should be based on various topics

Examples:
Do you know how to connect to Wi‑Fi from inside this bunker?
Can you tell me which social media platform you use to contact your suppliers?
    '''
    return prompt

def extract_dialogue_lines(text: str) -> Set[str]:
    dialogue_lines = set()

    lines = text.lstrip().rstrip().split('\n')
    for line in lines:
        if line:
            line = re.sub(r"^\s*\d+\.\s*", "", line)
            dialogue_lines.add(line.lstrip().rstrip())

    return dialogue_lines

if __name__ == '__main__':
    QUESTIONS_PER_ITERATION = 100
    npc_desc = read_file(f'resources/{npc}/npc_description.md')
    helper = OllamaHelper(OLLAMA_HOST)
    prompt = make_negative_prompt(npc_desc, QUESTIONS_PER_ITERATION)

    questions: Set[str] = set()

    while len(questions) < QUESTIONS_PER_ITERATION * 10:
        questions_str, think = helper.generate(MODEL, prompt)
        questions.update(extract_dialogue_lines(questions_str))

    # request response pairs
    rrps = list()

    for q in questions:
        npc_callback_action = Action.parse_action('DoNothing')
        request = UserRequest(
            context='You are asked a question to which you cannot know the answer.',
            state_of_user='',
            request_of_user=q,
        )
        response = NpcResponse(
            emotion='',
            answer='',
            action=npc_callback_action,
        )

        rrp = RequestResponsePair(user_request=request, npc_response=response)

        rrps.append(rrp)

    save_dataclass_records_to_jsonl(rrps, output_file=f'resources/{npc}/output/2_generated_irrelevant_player_requests.json')

    print('end of generating irrelevant questions')