import json
import os

from common.constants import DATA_DIR_NAME


def generate_validation_report(
    manifest: dict,
    metrics_base: dict,
    metrics_lora: dict,
    title: str = "LoRA Validation Report",
) -> str:
    """
    Builds a Markdown validation report comparing base model vs LoRA adapter.

    Works for both the LLM LoRA metrics (with json/action/args fail keys and a
    per-action-args table) and the embedding LoRA metrics (which only carry
    total_fails / fails_per_action). Rows and tables are rendered only when the
    corresponding key is present in the metrics, so the same function serves
    both pipelines.

    Parameters
    ----------
    manifest      : parsed manifest.json dict
    metrics_base  : metrics dict for the base model
    metrics_lora  : metrics dict for the LoRA fine-tuned model
    title         : report heading (e.g. "LoRA Validation Report" or
                    "Embedding LoRA Validation Report")

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

    def summary_row(label: str, key: str) -> str | None:
        """Render a summary row only if the metric key exists in the metrics."""
        if key not in metrics_base and key not in metrics_lora:
            return None
        b = metrics_base.get(key, 0)
        l = metrics_lora.get(key, 0)
        return (
            f"| {label} "
            f"| {b} ({pct(b, total)}) "
            f"| {l} ({pct(l, total)}) "
            f"| {delta_str(b, l, total)} |"
        )

    lines += [
        f"# {title}",
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
    ]

    summary_specs = [
        ("**Total fails**", "total_fails"),
        ("JSON parse fails", "json_parse_fails"),
        ("JSON structure fails", "json_structure_fails"),
        ("Actions fails", "total_action_fails"),
        ("Actions args fails", "total_args_fails"),
    ]
    lines += [row for label, key in summary_specs
              if (row := summary_row(label, key)) is not None]

    if "fails_per_action" in metrics_base or "fails_per_action" in metrics_lora:
        lines += [
            f"",
            f"---",
            f"",
            f"## Action Fails",
            f"",
            dict_table(metrics_base.get("fails_per_action", {}),
                       metrics_lora.get("fails_per_action", {}), total),
        ]

    if "fails_per_action_args" in metrics_base or "fails_per_action_args" in metrics_lora:
        lines += [
            f"",
            f"---",
            f"",
            f"## Args Fails",
            f"",
            dict_table(metrics_base.get("fails_per_action_args", {}),
                       metrics_lora.get("fails_per_action_args", {}), total),
        ]

    lines += [f""]

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
        "total_action_fails": 3,
        "total_args_fails": 3,

        "fails_per_action": {"buy": 2, "sell": 1},
        "fails_per_action_args": {"gold": 1, "item": 2},
    }

    metrics_lora_model = {
        "total_fails": 7,
        "total_requests": 10,

        "json_parse_fails": 2,
        "json_structure_fails": 1,
        "total_action_fails": 1,
        "total_args_fails": 2,

        "fails_per_action": {"buy": 1, "sell": 0},
        "fails_per_action_args": {"gold": 1, "item": 1},
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