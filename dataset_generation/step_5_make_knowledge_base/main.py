import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from common.constants import DATA_DIR_NAME, GEN_NPC_ANSWER_DIR_NAME, SENTENCE2_MODE_USER_REQUEST
from common.helpers import (
    format_action_signature,
    get_npc_data,
    parse_action_signature,
    save_text_file,
)
from common.manifest import Manifest


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


def process(
    lora_dir_path: str,
):
    manifest_f_path = os.path.abspath(f'{lora_dir_path}/manifest.json')
    manifest = Manifest(manifest_f_path)

    # temporary solution
    black_list = os.getenv('STEP_4E_BLACK_LIST_FOR_DIALOGS_PER_ACTION', '').split(',')

    # NOTE: STEP_5_MAX_RECORDS_PER_ACTION is obsolete since the KB is now
    # deduplicated by (action_name, parameters). One chunk per variant.
    npc_data = get_npc_data(manifest.unreal_commit(), manifest.npc_name(), manifest.flow_run_id())

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

    # Knowledge base is a deduplicated list of unique (action, parameters)
    # variants. Each chunk's text is the canonical action signature — same
    # format the embedding LoRA was trained against (sentence2 in
    # action_signature mode). This keeps retrieval consistent with training:
    # user request -> nearest action signature.
    knowledge_base = list()
    seen_keys: set[str] = set()

    for action_name in actions_set:
        dialogs_per_action_dir_path = f'{DATA_DIR_NAME}/{manifest.unreal_commit()}/{manifest.npc_name()}/{manifest.flow_run_id()}/{GEN_NPC_ANSWER_DIR_NAME}/{action_name}.jsonl'

        groups = group_by_action(dialogs_per_action_dir_path)
        for group_key, data_lst in groups.items():
            if group_key in seen_keys or not data_lst:
                continue

            first = data_lst[0]

            npc_response = first['npc_response']
            usr_request = first['usr_request']['request']
            action_obj = npc_response['action']

            if manifest.emb_dataset_mode() == SENTENCE2_MODE_USER_REQUEST:
                signature = usr_request
            else:
                signature = format_action_signature(
                    action_obj.get('name', action_name),
                    action_obj.get('parameters', {}) or {},
                )

            record = {
                'signature': signature,
                'action': action_obj,
            }
            knowledge_base.append(record)
            seen_keys.add(group_key)

    knowledge_base_str = json.dumps(knowledge_base, ensure_ascii=False, indent=2)
    save_text_file(
        folder_path=f"{lora_dir_path}",
        filename=f"knowledge_base.json",
        content=knowledge_base_str
    )


if __name__ == "__main__":
    hash = 'f71e60c'
    hash = '3d1c75f'
    lora_path = f'input_data/7c01ee7/trader/v2/training/lora_embedding/BAAI/bge-base-en-v1.5/user_request/{hash}'
    process(lora_path)
    lora_path = f'input_data/7c01ee7/trader/v2/training/lora_embedding/BAAI/bge-base-en-v1.5/action_signature/{hash}'
    process(lora_path)