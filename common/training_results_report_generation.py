import json
import os

from common.constants import DATA_DIR_NAME


def generate_validation_report(
    manifest: dict,
    metrics_base: dict,
    metrics_lora: dict,
) -> str:
    """
    Builds a Markdown validation report comparing base model vs LoRA adapter.

    Parameters
    ----------
    manifest      : parsed manifest.json dict
    metrics_base  : metrics dict for the base model
    metrics_lora  : metrics dict for the LoRA fine-tuned model

    Returns
    -------
    str – full Markdown text, ready to be written to a .md file
    """

    def pct(fails: int, total: int) -> str:
        if total == 0:
            return "—"
        return f"{fails / total * 100:.1f}%"

    def delta_str(base_val: int, lora_val: int, total: int) -> str:
        """Shows absolute and relative change; green arrow if improved."""
        diff = lora_val - base_val
        diff_pct = (diff / total * 100) if total else 0
        sign = "+" if diff > 0 else ""
        arrow = "▼" if diff < 0 else ("▲" if diff > 0 else "—")
        return f"{arrow} {sign}{diff} ({sign}{diff_pct:.1f}%)"

    def dict_table(base_d: dict, lora_d: dict, total: int) -> str:
        all_keys = sorted(set(base_d) | set(lora_d))
        rows = ["| Action / Key | Base | LoRA | Δ |",
                "|---|---:|---:|---|"]
        for k in all_keys:
            b = base_d.get(k, 0)
            l = lora_d.get(k, 0)
            rows.append(
                f"| `{k}` "
                f"| {b} ({pct(b, total)}) "
                f"| {l} ({pct(l, total)}) "
                f"| {delta_str(b, l, total)} |"
            )
        return "\n".join(rows)

    total = metrics_base["total_requests"]

    # ── Dataset summary from manifest ──────────────────────────────────
    ds = manifest.get("dataset", {})
    ds_train = ds.get("training", {})
    ds_val   = ds.get("validation", {})

    def actions_inline(actions: dict) -> str:
        return ", ".join(f"`{k}`: {v}" for k, v in actions.items())

    # ── Build report ───────────────────────────────────────────────────
    lines: list[str] = []

    lines += [
        f"# LoRA Validation Report",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **Timestamp** | {manifest.get('timestamp', '—')} |",
        f"| **Unreal commit** | `{manifest.get('unreal_commit', '—')}` |",
        f"| **NPC** | `{manifest.get('npc_name', '—')}` |",
        f"| **Pipeline commit** | `{manifest.get('pipeline_commit', '—')}` |",
        f"| **Flow run id** | `{manifest.get('flow_run_id', '—')}` |",
        f"",
        f"---",
        f"",
        f"## Dataset",
        f"",
        f"| Split | Actions | Total |",
        f"|---|---|---:|",
        f"| Training   | {actions_inline(ds_train.get('actions', {}))} | **{ds_train.get('total', 0)}** |",
        f"| Validation | {actions_inline(ds_val.get('actions', {}))} | **{ds_val.get('total', 0)}** |",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"> Validation set: **{total}** requests",
        f"",
        f"| Metric | Base model | LoRA model | Δ |",
        f"|---|---:|---:|---|",
        f"| **Total fails** "
            f"| {metrics_base['total_fails']} ({pct(metrics_base['total_fails'], total)}) "
            f"| {metrics_lora['total_fails']} ({pct(metrics_lora['total_fails'], total)}) "
            f"| {delta_str(metrics_base['total_fails'], metrics_lora['total_fails'], total)} |",
        f"| JSON parse fails "
            f"| {metrics_base['json_parse_fails']} ({pct(metrics_base['json_parse_fails'], total)}) "
            f"| {metrics_lora['json_parse_fails']} ({pct(metrics_lora['json_parse_fails'], total)}) "
            f"| {delta_str(metrics_base['json_parse_fails'], metrics_lora['json_parse_fails'], total)} |",
        f"| JSON structure fails "
            f"| {metrics_base['json_structure_fails']} ({pct(metrics_base['json_structure_fails'], total)}) "
            f"| {metrics_lora['json_structure_fails']} ({pct(metrics_lora['json_structure_fails'], total)}) "
            f"| {delta_str(metrics_base['json_structure_fails'], metrics_lora['json_structure_fails'], total)} |",
        f"",
        f"---",
        f"",
        f"## Action Fails",
        f"",
        dict_table(metrics_base["action_fails"], metrics_lora["action_fails"], total),
        f"",
        f"---",
        f"",
        f"## Args Fails",
        f"",
        dict_table(metrics_base["args_fails"], metrics_lora["args_fails"], total),
        f"",
    ]

    return "\n".join(lines)

if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v_test'

    metrics_base_model = {
        "total_fails": 10,
        "total_requests": 10,
        "json_parse_fails": 4,
        "json_structure_fails": 3,
        "action_fails": {"buy": 2, "sell": 1},
        "args_fails": {"gold": 1, "item": 2},
    }

    metrics_lora_model = {
        "total_fails": 7,
        "total_requests": 10,
        "json_parse_fails": 2,
        "json_structure_fails": 1,
        "action_fails": {"buy": 1, "sell": 0},
        "args_fails": {"gold": 1, "item": 1},
    }

    flow_run_dir_path = f'{DATA_DIR_NAME}/{COMMIT}/{NPC_NAME}/{FLOW_RUN_ID}'

    with open(os.path.join(flow_run_dir_path, 'manifest.json'), 'r') as f:
        manifest_dict = json.loads(f.read())
        md_report = generate_validation_report(
            manifest=manifest_dict,
            metrics_base=metrics_base_model,
            metrics_lora=metrics_lora_model,
        )

    with open(os.path.join(flow_run_dir_path, 'report.md'), 'w', encoding="utf-8") as f:
        f.write(md_report)

    print()