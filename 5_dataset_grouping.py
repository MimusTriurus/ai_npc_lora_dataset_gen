import json
import os
import random
from collections import defaultdict

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

def normalize_dataset(data_files: str, dataset_group_size: int = -1) -> Dataset:
    dataset: Dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )

    def extract_action_name(example):
        data = json.loads(example["messages"][-1]["content"])
        return {"action_name": data["action"]["name"]}

    dataset = dataset.map(extract_action_name)

    indices_by_action = defaultdict(list)

    for idx, example in enumerate(dataset):
        indices_by_action[example["action_name"]].append(idx)

    sizes = {k: len(v) for k, v in indices_by_action.items()}
    print("Sizes data by Action class:", sizes)

    target_size = dataset_group_size if dataset_group_size != -1 else int(sum(sizes.values()) / len(sizes))

    print("Target size for each Action class:", target_size)

    balanced_groups = {}

    for action_name, idxs in indices_by_action.items():
        current_size = len(idxs)

        if current_size > target_size:
            # UNDERSAMPLING
            new_idxs = random.sample(idxs, target_size)

        elif current_size < target_size:
            # OVERSAMPLING
            extra = random.choices(idxs, k=target_size - current_size)
            new_idxs = idxs + extra

        else:
            new_idxs = idxs

        balanced_groups[action_name] = dataset.select(new_idxs)

    balanced_dataset = DatasetDict(balanced_groups)

    merged_dataset = concatenate_datasets(list(balanced_dataset.values()))

    return merged_dataset

if __name__ == "__main__":
    result = normalize_dataset('resources/npc_trader/output/instruct_dataset/training/*.jsonl', 500)
    print(result)
