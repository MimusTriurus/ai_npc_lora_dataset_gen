import json
import os
from prefect import task
from common.constants import DATA_DIR_NAME, GGUF_DIR_NAME
from common.helpers import is_env_var_true
from common.storage import MinioStorage

from prefect.artifacts import create_link_artifact, create_markdown_artifact, create_image_artifact


@task(name="step_3_save-artifacts")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
    S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
    S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
    S3_IS_SECURE = is_env_var_true("S3_IS_SECURE")

    storage = MinioStorage(
        endpoint=S3_ENDPOINT,
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        bucket="interactive-npc-lora-models",
        secure=S3_IS_SECURE,
    )

    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = f'{flow_run_dir_path}/manifest.json'
    with open(manifest_f_path, "r", encoding="utf-8", errors="replace") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

        key = f'{flow_run_dir_path}/{model_name}_lora_f16.gguf'
        path = lora_adapter_f_path
        link_lora = storage.upload_file(key=key, path=path)
        storage.upload_file(key=manifest_f_path, path=manifest_f_path)

        create_link_artifact(
            key="lora-model",
            link=link_lora,
            description=f"{npc_name} lora adapter (.gguf)",
        )

        key = f'{flow_run_dir_path}/report.md'
        path = f'{flow_run_dir_path}/reports/report.md'
        storage.upload_file(key=key, path=path)

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            markdown_report = f.read()

            create_markdown_artifact(
                key="lora-training-report",
                markdown=markdown_report,
                description="NPC LoRA adapter training report",
            )

        key = f'{flow_run_dir_path}/agg_metrics_chart.png'
        path = f'{flow_run_dir_path}/reports/agg_metrics_chart.png'
        link_chart_agg_metrics = storage.upload_file(key=key, path=path)

        create_image_artifact(
            image_url=link_chart_agg_metrics,
            description="agg_metrics_chart",
            key="agg-metrics-chart"
        )

        key = f'{flow_run_dir_path}/actions_metrics_chart.png'
        path = f'{flow_run_dir_path}/reports/actions_metrics_chart.png'
        link_chart_actions_metrics = storage.upload_file(key=key, path=path)
        create_image_artifact(
            image_url=link_chart_actions_metrics,
            description="actions_metrics_chart",
            key="actions-metrics-chart"
        )

        key=f'{flow_run_dir_path}/actions_args_metrics_chart.png'
        path = f'{flow_run_dir_path}/reports/actions_args_metrics_chart.png'
        link_chart_args_metrics = storage.upload_file(key=key, path=path)
        create_image_artifact(
            image_url=link_chart_args_metrics,
            description="actions_args_metrics_chart",
            key="actions-args-metrics-chart"
        )

if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'

    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)