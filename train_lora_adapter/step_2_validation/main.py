import json
import os
from common.constants import *
from prefect import task
from pathlib import Path
from common.helpers import update_manifest, read_dataset_file
from common.inference import inference, get_system_prompt, make_ullama_config
from common.manifest import Manifest
from common.metrics_plot_generation import make_metrics_plot
from common.ollama_helper import OllamaHelper, MODEL
from common.training_results_report_generation import generate_validation_report
from common.ullama_helper import ULlamaHelper

@task(name="step_2_emb_lora_validation")
def process(model_dir_path: str):
    manifest_f_path = os.path.abspath(f'{model_dir_path}/manifest.json')
    manifest = Manifest(manifest_f_path)
    ullama_inference_cfg = json.loads(open(f'{model_dir_path}/inference_cfg.json', "r", encoding="utf-8").read())

    print(f'===> Lora model inference')

    current_dir = Path.cwd()
    print(f'Current dir: {current_dir}')

    llm_model_f_path = ullama_inference_cfg.get('model', '')
    llm_model_f_path = Path.joinpath(current_dir,llm_model_f_path).resolve().as_posix()
    if not os.path.isfile(llm_model_f_path):
        print(f"[ERROR] LLM model not found {llm_model_f_path}")
        exit(1)

    lora_adapter_f_path = ullama_inference_cfg.get('lora_adapter', '')
    lora_adapter_f_path = Path.joinpath(current_dir, lora_adapter_f_path).resolve().as_posix()
    if not os.path.isfile(lora_adapter_f_path):
        print(f"[Warning] LLM LoRA adapter not found {lora_adapter_f_path}")
        ullama_inference_cfg['lora_adapter'] = ''

    validation_dataset_dir_path = f'{manifest.dataset_d_path()}/validation'

    lora_metrics = inference(
        inference_config=ullama_inference_cfg,
        inference_type=ULlamaHelper,
        validation_dataset_dir_path=validation_dataset_dir_path
    )
    ullama_inference_cfg['lora_adapter'] = ''

    base_metrics = inference(
        inference_config=ullama_inference_cfg,
        inference_type=ULlamaHelper,
        validation_dataset_dir_path=validation_dataset_dir_path
    )

    validation_data = {
        "validation" : {
            "base_metrics": base_metrics,
            "lora_metrics": lora_metrics
        }
    }

    update_manifest(manifest_f_path, validation_data)

    Path(f"{model_dir_path}/reports/").mkdir(parents=True, exist_ok=True)

# region make .md report
    md_report = generate_validation_report(
        manifest=manifest.to_dict(),
        metrics_base=base_metrics,
        metrics_lora=lora_metrics,
    )

    with open(os.path.join(f'{model_dir_path}/reports/', 'report.md'), 'w', encoding="utf-8") as f:
        f.write(md_report)
# endregion

# region make metrics plots
    make_metrics_plot(
        metrics_model_base=base_metrics,
        metrics_model_lora=lora_metrics,
        lora_dir_path=model_dir_path
    )
# endregion

if __name__ == "__main__":
    unreal_hash = os.getenv('COMMIT')
    npc_name = os.getenv('NPC_NAME')
    flow_run_id = os.getenv('FLOW_RUN_ID')

    llm_model = os.getenv('STEP_0_MODEL_NAME')
    llm_hash = os.getenv('LLM_TRAINING_SESSION_HASH')
    lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora/{llm_model}/chat/{llm_hash}'
    if os.path.isdir(lora_path):
        process(lora_path)
    else:
        print(f"===> Error: can't find lora llm: {lora_path}")
    exit()
    lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora/{llm_model}/tool_calling/{llm_hash}'
    if os.path.isdir(lora_path):
        process(lora_path)
    else:
        print(f"===> Error: can't find lora llm: {lora_path}")
