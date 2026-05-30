import glob
import json
import os
from pathlib import Path

from prefect import task
from common.constants import DATA_DIR_NAME, GGUF_DIR_NAME
from common.helpers import is_env_var_true
from common.storage import MinioStorage

from prefect.artifacts import create_link_artifact, create_markdown_artifact, create_image_artifact


@task(name="step_3_save-artifacts")
def process(model_dir_path: str):
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

    flow_run_dir_path = f'{model_dir_path}'
    manifest_f_path = f'{flow_run_dir_path}/manifest.json'
    with open(manifest_f_path, "r", encoding="utf-8", errors="replace") as f:
        #manifest = json.load(f)
        #model_name = manifest["training"]["model_name"].lower()
        lora_adapter_f_path = f'{flow_run_dir_path}/lora.gguf'
        #key = f'{flow_run_dir_path}/{model_name}_lora_f16.gguf'
        key = lora_adapter_f_path
        path = lora_adapter_f_path
        link_lora = storage.upload_file(key=key, path=path)
        storage.upload_file(key=manifest_f_path, path=manifest_f_path)

        create_link_artifact(
            key="lora-model",
            link=link_lora,
            description=f"{npc_name} lora adapter (.gguf)",
        )

# region create .MD report
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
# endregion

# region upload CHARTS
        reports_dir = f'{flow_run_dir_path}/reports'
        for png_path in sorted(glob.glob(f'{reports_dir}/*.png')):
            filename = os.path.basename(png_path)
            name = os.path.splitext(filename)[0]
            key = f'{flow_run_dir_path}/{filename}'
            link_chart = storage.upload_file(key=key, path=png_path)
            create_image_artifact(
                image_url=link_chart,
                description=name,
                key=name.replace('_', '-').lower(),
            )
# endregion

# region upload llm config
        key = f'{flow_run_dir_path}/inference_cfg.json'
        path = key
        storage.upload_file(key=key, path=path)
# endregion

# region knowledge_base
        key = f'{flow_run_dir_path}/knowledge_base.json'
        if Path(key).is_file():
            path = key
            storage.upload_file(key=key, path=path)
# endregion




if __name__ == "__main__":
    unreal_hash = os.getenv('COMMIT')
    npc_name = os.getenv('NPC_NAME')
    flow_run_id = os.getenv('FLOW_RUN_ID')

    llm_model = os.getenv('STEP_0_MODEL_NAME')
    llm_hash = os.getenv('LLM_TRAINING_SESSION_HASH')
    llm_lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora/{llm_model}/chat/{llm_hash}'

    emb_model = os.getenv('STEP_0_EMB_MODEL_NAME')
    emb_hash = os.getenv('EMB_TRAINING_SESSION_HASH')
    emb_lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora_embedding/{emb_model}/action_signature/{emb_hash}'

    process(model_dir_path=llm_lora_path)
    process(model_dir_path=emb_lora_path)