import ollama
import re
from typing import Optional, List
import random
import os

emotions =['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise']
DEFAULT_NPC_DESCRIPTION = f'''Name: Kaelen Swiftarrow
Race: Half-Elf
Specialization: Ranger / Beastmaster
Background: A frontier outcast who found kinship with a wolf companion. Now a silent guardian of the wilds.
Character Traits: Loner, distrustful-of-civilization, protective, dry-wit, stern.
'''

MODEL = os.getenv('MODEL', 'qwen3:8b')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
PARTIAL_DATASET_SIZE = os.getenv('PARTIAL_DATASET_SIZE', 50)
DATASET_SIZE = os.getenv('DATASET_SIZE', 500)
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'npc_lora_dataset.jsonl')
NEGATIVE_ITERATIONS_COUNT = os.getenv('NEGATIVE_ITERATIONS_COUNT', 5)

NPC_DESCRIPTION = os.getenv('NPC_DESCRIPTION', DEFAULT_NPC_DESCRIPTION)

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

def make_negative_prompt(npc_description: str, records_count: int) -> str:
    prompt = f'''
    NPC: {npc_description}.
    Generate deliberately irrelevant questions (things that the NPC described above cannot know, such as what the internet is or whether he knows that he is an NPC) for the NPC described above. 
    The NPC should respond to such questions with anger or confusion (not understanding what is being asked), but in accordance with his role and character traits.
    The NPC should not assume what the player means, but immediately respond that they do not understand what is being said.
    Each entry format:
    <number>|<player question>|<emotion>|<NPC response>
    Allowed emotions: [Angry], [Surprise].
    Generate {records_count} lines, with many questions deliberately outside the NPC’s domain.
    '''

    return prompt

class OllamaDialogueExtractor:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)

    def send_request(self, model: str, prompt: str) -> Optional[str]:
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
    with open(filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    return True

def make_dataset_using_ollama(emotion: str) -> list:
    extractor = OllamaDialogueExtractor(OLLAMA_HOST)
    prompt = make_prompt(NPC_DESCRIPTION, emotion, PARTIAL_DATASET_SIZE)
    response = extractor.send_request(MODEL, prompt)

    if response is None:
        print("Can't get a response from Ollama")
        return []
    dialogue_lines = extract_dialogue_lines(response)

    if not dialogue_lines:
        print("Dataset is empty")
        print("\nFull Ollama response:")
        print(response)
        return []
    return dialogue_lines

def make_negative_dataset_using_ollama() -> list:
    extractor = OllamaDialogueExtractor(OLLAMA_HOST)
    prompt = make_negative_prompt(NPC_DESCRIPTION, PARTIAL_DATASET_SIZE)
    response = extractor.send_request(MODEL, prompt)

    if response is None:
        print("Can't get a response from Ollama")
        return []
    dialogue_lines = extract_dialogue_lines(response)

    if not dialogue_lines:
        print("Dataset is empty")
        print("\nFull Ollama response:")
        print(response)
        return []
    return dialogue_lines

def main():
    print(f"= Model: {MODEL}")
    print(f"= Ollama host: {OLLAMA_HOST}")

    dataset = set()
    print('= Generate negative requests')
    for i in range(NEGATIVE_ITERATIONS_COUNT):
        print(f'== {i}/{NEGATIVE_ITERATIONS_COUNT - 1}')
        negative_dataset = make_negative_dataset_using_ollama()
        dataset.update(negative_dataset)

    for emotion in emotions:
        print(f'== Generate dataset for emotion: {emotion}')
        emotion_dataset = set()
        while len(emotion_dataset) < DATASET_SIZE:
            print('=== Request processing....')
            emotion_dataset.update(make_dataset_using_ollama(emotion))
            print(f"=== Emotions dataset current size {len(emotion_dataset)}")
        dataset.update(emotion_dataset)
        print(f"== Full dataset current size {len(dataset)}")
    if not dataset:
        print(f'= Dataset is empty!')
        return
    print(f"= Full dataset final size {len(dataset)}")

    shuffled_list = list(dataset)
    random.shuffle(shuffled_list)

    if save_to_file(shuffled_list, OUTPUT_FILE):
        print(f"\n= Dataset saved into: '{OUTPUT_FILE}'")
    else:
        print("\n= Error on file save")


if __name__ == "__main__":
    main()