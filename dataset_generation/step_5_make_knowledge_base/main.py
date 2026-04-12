import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from common.constants import DATA_DIR_NAME, GEN_NPC_ANSWER_DIR_NAME
from common.helpers import get_npc_data, parse_action_signature, save_text_file


def make_action_key(action: dict) -> str:
    name = action.get("name", "")
    params = action.get("parameters", {})
    params_str = json.dumps(params, ensure_ascii=False, sort_keys=True)
    return f"{name}::{params_str}"


def group_by_action(input_path: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [!] Строка {line_num}: ошибка парсинга — {e}", file=sys.stderr)
                continue

            action = record.get("npc_response", {}).get("action", {})
            if not action:
                key = "NO_ACTION::null"
            else:
                key = make_action_key(action)

            groups[key].append(record)

    return dict(groups)


def save_groups(groups: dict[str, list[dict]], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}

    for key, records in groups.items():
        # Безопасное имя файла
        safe_name = key.replace("::", "__").replace("/", "-").replace(" ", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_.")
        file_path = out / f"{safe_name}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary[key] = {"count": len(records), "file": str(file_path)}
        print(f"  [{len(records):>4} записей]  {key}")

    # Сохраняем сводку
    summary_path = out / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nСводка сохранена: {summary_path}")


def process(git_commit: str, npc_name: str, flow_run_id: str):
    # temporary solution
    black_list = [
        'NotEnoughGoldToBuy',
        'OutOfStock',
        'DoNothing',
    ]

    MAX_RECORDS_PER_ACTION = int(os.getenv("STEP_5_MAX_RECORDS_PER_ACTION", 2))

    npc_data = get_npc_data(git_commit, npc_name, flow_run_id)

    actions_template_data = npc_data['ActionData']

    actions_set = set()

    for action_template_data in actions_template_data:
        has_data_getter = action_template_data.get('HasDataGetter', 'true') in ("1", "true", "yes", "on")
        if not has_data_getter:
            continue
        action_name, arg_names = parse_action_signature(action_template_data['ActionTemplate'])
        if action_name in black_list:
            continue
        actions_set.add(action_name)

    knowledge_base = list()

    for action_name in actions_set:
        dialogs_per_action_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_NPC_ANSWER_DIR_NAME}/{action_name}.jsonl'

        groups = group_by_action(dialogs_per_action_dir_path)
        for group_key, data_lst in groups.items():
            size = min(len(data_lst), MAX_RECORDS_PER_ACTION)
            for data_dict in data_lst[:size]:
                record = {
                    'request': data_dict['usr_request']['request'],
                    'action': data_dict['npc_response']['action'],
                }
                knowledge_base.append(record)

    knowledge_base_str = json.dumps(knowledge_base, ensure_ascii=False, indent=2)
    save_text_file(
        folder_path=f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}",
        filename=f"knowledge_base.json",
        content=knowledge_base_str
    )


if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID))