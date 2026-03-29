import io
import base64
import matplotlib.pyplot as plt

from common.constants import DATA_DIR_NAME

#from prefect.artifacts import create_artifact

def compare_two_models_metrics(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    title: str = '',
    total_requests: int = 10,
):
    flat_a = metrics_a
    flat_b = metrics_b

    categories = list(flat_a.keys())
    values_a = [flat_a[c] for c in categories]
    values_b = [flat_b[c] for c in categories]

    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars_a = plt.bar([i - width/2 for i in x], values_a, width=width, label=label_a)
    bars_b = plt.bar([i + width/2 for i in x], values_b, width=width, label=label_b)

    for bar in bars_a:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom'
        )

    for bar in bars_b:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(height),
            ha='center',
            va='bottom'
        )

    plt.xticks(x, categories, rotation=30)
    plt.ylabel("Count")

    step = min(5, total_requests)
    plt.yticks(
        ticks=[i for i in range(total_requests) if i % step == 0],
    )

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    return plt

def metrics_agg(m):
    return {
        "total_fails": m["total_fails"],
        "json_parse_fails": m["json_parse_fails"],
        "json_structure_fails": m["json_structure_fails"],
        "action_fails": m['total_action_fails'],
        "args_fails": m['total_args_fails'],
    }

def make_metrics_plot(
    git_commit: str, npc_name: str, flow_run_id: str,
    metrics_model_base: dict,
    metrics_model_lora: dict,
):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    total_requests = metrics_model_lora["total_requests"]
    # --------------------------------
    plt = compare_two_models_metrics(
        metrics_agg(metrics_model_base),
        metrics_agg(metrics_model_lora),
        "Base",
        "LoRA",
        "Aggregated metrics comparison",
        total_requests
    )
    plt.savefig(f"{flow_run_dir_path}/reports/agg_metrics_chart.png", dpi=200, bbox_inches="tight")
    plt.close()
# --------------------------------
    plt = compare_two_models_metrics(
        metrics_model_base["fails_per_action"],
        metrics_model_lora["fails_per_action"],
        "Base",
        "LoRA",
        "Failed actions metrics comparison",
        total_requests
    )
    plt.savefig(f"{flow_run_dir_path}/reports/actions_metrics_chart.png", dpi=200, bbox_inches="tight")
    plt.close()
# --------------------------------
    plt = compare_two_models_metrics(
        metrics_model_base["fails_per_action_args"],
        metrics_model_lora["fails_per_action_args"],
        "Base",
        "LoRA",
        "Failed actions args metrics comparison",
        total_requests
    )
    plt.savefig(f"{flow_run_dir_path}/reports/actions_args_metrics_chart.png", dpi=200, bbox_inches="tight")
    plt.close()

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

    make_metrics_plot(
        COMMIT,
        NPC_NAME,
        FLOW_RUN_ID,
        metrics_base_model,
        metrics_lora_model,
    )
