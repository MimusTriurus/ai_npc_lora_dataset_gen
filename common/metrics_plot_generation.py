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


    buf = io.BytesIO()
    plt.savefig(f"{flow_run_dir_path}/metrics_chart.png", dpi=200, bbox_inches="tight")

    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return
    # --- артефакт Prefect ---
    create_artifact(
        key="llm_metrics_comparison",
        type="image",
        description="Comparison of two LLM inference metric sets",
        data=base64.b64encode(buf.read()).decode("utf-8"),
    )

def metrics_agg(m):
    return {
        "total_fails": m["total_fails"],
        "json_parse_fails": m["json_parse_fails"],
        "json_structure_fails": m["json_structure_fails"],
        "action_fails": sum(m["action_fails"].values()),
        "args_fails": sum(m["args_fails"].values()),
    }

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

    total_requests = metrics_lora_model["total_requests"]

    flow_run_dir_path = f'{DATA_DIR_NAME}/{COMMIT}/{NPC_NAME}/{FLOW_RUN_ID}'

    plt = compare_two_models_metrics(
        metrics_agg(metrics_base_model),
        metrics_agg(metrics_lora_model),
        "Base",
        "LoRA",
        "Aggregated metrics comparison",
        total_requests
    )

    plt.savefig(f"{flow_run_dir_path}/agg_metrics_chart.png", dpi=200, bbox_inches="tight")

    plt = compare_two_models_metrics(
        metrics_base_model["action_fails"],
        metrics_lora_model["action_fails"],
        "Base",
        "LoRA",
        "Failed actions metrics comparison",
        total_requests
    )

    plt.savefig(f"{flow_run_dir_path}/actions_metrics_chart.png", dpi=200, bbox_inches="tight")

    plt = compare_two_models_metrics(
        metrics_base_model["args_fails"],
        metrics_lora_model["args_fails"],
        "Base",
        "LoRA",
        "Failed actions args metrics comparison",
        total_requests
    )

    plt.savefig(f"{flow_run_dir_path}/actions_args_metrics_chart.png", dpi=200, bbox_inches="tight")
