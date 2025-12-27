import random

from common.helpers import load_jsonl_to_dataclasses, read_file, save_dataclass_records_to_jsonl, \
    save_dict_records_to_jsonl
from common.data_classes import *

'''
{
  "messages": [
    {
      "role": "system",
      "content": "You are a proud forest guardian. You speak sharply, independently, and with dignity."
    },
    {
      "role": "user",
      "content": "You’ve called me a guardian, yet you’ve demanded I swear loyalty to a king."
    },
    {
      "role": "assistant",
      "content": "[Surprise] A king? I exhaled sharply. You’ve no right to command a guardian. The wilds are not your kingdom."
    }
  ]
}
'''

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

    irrelevant_rr = load_jsonl_to_dataclasses('resources/npc_trader/output/3_generated_irrelevant_requests_responses.json', RequestResponsePair)
    relevant_rr = load_jsonl_to_dataclasses('resources/npc_trader/output/3_generated_relevant_requests_responses.json', RequestResponsePair)

    rrs = irrelevant_rr + relevant_rr
    random.shuffle(rrs)

    dataset = []

    for rr in rrs:
        if not rr.npc_response['answer']:
            continue

        usr_content = json.dumps(rr.user_request)
        usr_request = {
            "role": "user",
            "content": usr_content
        }

        npc_content = json.dumps(rr.npc_response)
        npc_response = {
            "role": "assistant",
            "content": npc_content
        }
        record = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
        }
        record["messages"].append(usr_request)
        record["messages"].append(npc_response)

        dataset.append(record)

    random.shuffle(dataset)

    n = int(len(dataset) * 0.9)
    training_dataset = dataset[:n]
    validation_dataset = dataset[n:]

    save_dict_records_to_jsonl(training_dataset, 'resources/npc_trader/output/4_training_dataset.jsonl')
    save_dict_records_to_jsonl(validation_dataset, 'resources/npc_trader/output/4_validation_dataset.jsonl')

    print()