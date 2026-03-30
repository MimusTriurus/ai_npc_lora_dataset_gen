import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.constants import DATA_DIR_NAME


def compare_two_models_metrics(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    title: str = "",
    total_requests: int = 10,
) -> plt.Figure:
    categories = list(metrics_a.keys())
    values_a = [metrics_a[c] for c in categories]
    values_b = [metrics_b[c] for c in categories]

    x = range(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_a = ax.bar([i - width / 2 for i in x], values_a, width=width, label=label_a)
    bars_b = ax.bar([i + width / 2 for i in x], values_b, width=width, label=label_b)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, str(h), ha="center", va="bottom")

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, str(h), ha="center", va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=30)
    ax.set_ylabel("Count")

    step = min(5, total_requests)
    ax.set_yticks([i for i in range(total_requests) if i % step == 0])

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


def metrics_agg(m: dict) -> dict:
    return {
        "total_fails": m["total_fails"],
        "json_parse_fails": m["json_parse_fails"],
        "json_structure_fails": m["json_structure_fails"],
        "action_fails": m["total_action_fails"],
        "args_fails": m["total_args_fails"],
    }


def make_metrics_plot(
    git_commit: str,
    npc_name: str,
    flow_run_id: str,
    metrics_model_base: dict,
    metrics_model_lora: dict,
):
    flow_run_dir_path = f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}"
    total_requests = metrics_model_lora["total_requests"]

    charts = [
        ("agg_metrics_chart.png", metrics_agg(metrics_model_base), metrics_agg(metrics_model_lora), "Aggregated metrics comparison"),
        ("actions_metrics_chart.png", metrics_model_base["fails_per_action"], metrics_model_lora["fails_per_action"], "Failed actions metrics comparison"),
        ("actions_args_metrics_chart.png", metrics_model_base["fails_per_action_args"], metrics_model_lora["fails_per_action_args"], "Failed actions args metrics comparison"),
    ]

    for fname, m_base, m_lora, title in charts:
        fig = compare_two_models_metrics(m_base, m_lora, "Base", "LoRA", title, total_requests)
        fig.savefig(f"{flow_run_dir_path}/reports/{fname}", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    COMMIT = "7c01ee7d6b644dbf4d5ccc2b9c1db9adab96b34a"[:7]
    NPC_NAME = "trader"
    FLOW_RUN_ID = "v1"

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

    make_metrics_plot(
        COMMIT,
        NPC_NAME,
        FLOW_RUN_ID,
        metrics_base_model,
        metrics_lora_model,
    )