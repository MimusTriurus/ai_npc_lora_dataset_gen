import ollama
import re
from typing import Optional, List
from openai import OpenAI
import os

emotions =['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise']
MODEL = "qwen3:8b"
OLLAMA_HOST = "http://localhost:11434"
PARTIAL_DATASET_SIZE = 50
TARGET_EMOTION = 'surprise'
DATASET_SIZE = 500
OUTPUT_FILE = f"dataset_{TARGET_EMOTION}.jsonl"

GENERATION_PARAMS = {
    'temperature': 0.9,
    'top_p': 0.9,
    'top_k': 40
}

NPC_DESCRIPTION = f'''Name: Kaelen Swiftarrow
Race: Half-Elf
Specialization: Ranger / Beastmaster
Background: A frontier outcast who found kinship with a wolf companion. Now a silent guardian of the wilds.
Character Traits: Loner, distrustful-of-civilization, protective, dry-wit, stern.
'''

PROMPT = f"""NPC Description:
{NPC_DESCRIPTION}

Instruction:
Generate a dataset of {PARTIAL_DATASET_SIZE} interactions with the NPC described above.
For each interaction:
Create a user query designed to provoke one of the specified emotion {TARGET_EMOTION} in the NPC, based strictly on their background, race, and traits.

Generate the NPC's in-character response, accurately reflecting the targeted emotion - {TARGET_EMOTION}.
Format each entry strictly as:
<number>|<user query>|<emotion>|<NPC response>
The response should only contain the dataset without any extra information.
"""

def make_prompt(npc_description: str, target_emotion, records_count: int) -> str:
    prompt = f"""NPC Description:
    {npc_description}

    Instruction:
    Generate a dataset of {records_count} interactions with the NPC described above.
    For each interaction:
    Create a user query designed to provoke one of the specified emotion {target_emotion} in the NPC, based strictly on their background, race, and traits.

    Generate the NPC's in-character response, accurately reflecting the targeted emotion - {target_emotion}.
    Format each entry strictly as:
    <number>|<user query>|<emotion>|<NPC response>
    The response should only contain the dataset without any extra information.
    """
    return prompt

class OllamaDialogueExtractor:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)

    def send_request(self, model: str, prompt: str, **kwargs) -> Optional[str]:
        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                stream=False,
            )

            return response['response']

        except ollama.ResponseError as e:
            print(f"Error during Ollama generation process: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def check_model_exists(self, model: str) -> bool:
        try:
            models = self.client.list()
            model_names = [m['model'] for m in models['models']]
            return model in model_names
        except Exception as e:
            print(f"Ошибка при получении списка моделей: {e}")
            return False


def extract_dialogue_lines(text: str) -> List[str]:
    pattern = r'<think>.+?</think>(.+)'
    dialogue_lines = set()
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
    for match in matches:
        groups = match.lstrip().rstrip().split('\n')
        for group in groups:
            segments = group.split('|')
            if len(segments) >= 4:
                question = segments[1].replace('"', '')
                emotion = segments[2]
                answer = segments[3].replace('"', '')
                line = '{"instruction": "' + question + '", "output": "' + '[' + emotion + '] ' + answer + '"}'
                dialogue_lines.add(line)

    return list(dialogue_lines)


def save_to_file(lines: List[str], filename: str) -> bool:
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"Найдено и сохранено {len(lines)} строк в файл '{filename}'")
        return True
    except IOError as e:
        print(f"Ошибка при сохранении файла: {e}")
        return False

def make_dataset_using_ollama(emotion: str) -> list:
    extractor = OllamaDialogueExtractor(OLLAMA_HOST)
    prompt = make_prompt(NPC_DESCRIPTION, emotion, PARTIAL_DATASET_SIZE)
    response = extractor.send_request(MODEL, prompt, **GENERATION_PARAMS)

    if response is None:
        print("Can't get a response from Ollama")
        return []
    dialogue_lines = extract_dialogue_lines(response)

    if not dialogue_lines:
        print("Dataset is empty")
        print("\nFull Ollama response:")
        print(response)
        return []

    print(f"== Partial dataset size {len(dialogue_lines)}")
    return dialogue_lines

def main():
    print(f"Model: {MODEL}")
    print(f"Ollama host: {OLLAMA_HOST}")

    for emotion in emotions:
        dataset = set()
        print(f'== Generate dataset for emotion: {emotion}')
        while len(dataset) < DATASET_SIZE:
            print('== Request processing....')
            partial_dataset = make_dataset_using_ollama(emotion)
            dataset.update(partial_dataset)
            print(f"== Full dataset current size {len(dataset)}")

        print(f"== Full dataset final size {len(dataset)}")
        if save_to_file(list(dataset), OUTPUT_FILE):
            print(f"\nDataset saved into: '{OUTPUT_FILE}'")
        else:
            print("\nError on file save")


if __name__ == "__main__":
    main()