import json
import os
from common.constants import *
from prefect import task
from pathlib import Path
from common.helpers import update_manifest, read_dataset_file
from common.inference import inference, get_system_prompt, make_ullama_config
from common.metrics_plot_generation import make_metrics_plot
from common.ollama_helper import OllamaHelper, MODEL
from common.training_results_report_generation import generate_validation_report
from common.ullama_helper import ULlamaHelper

@task(name="step_2_lora_validation")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'
    manifest_f_path = os.path.abspath(f'{flow_run_dir_path}/manifest.json')

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    print(f'===> Base model inference')

    ollama_inference_cfg = {
        'model': MODEL,
        'system_prompt': get_system_prompt(f'{flow_run_dir_path}/{GEN_SYS_PROMPT_DIR_NAME}/system_prompt.txt'),
    }

    base_metrics = inference(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        inference_config=ollama_inference_cfg,
        inference_type=OllamaHelper
    )

    ullama_inference_cfg = make_ullama_config(git_commit, npc_name, flow_run_id, llm_model_f_path, lora_adapter_f_path)
    with open(os.path.join(f'{flow_run_dir_path}/', 'inference_cfg.json'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(ullama_inference_cfg, indent=2))

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

    lora_metrics = inference(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        inference_config=ullama_inference_cfg,
        inference_type=ULlamaHelper
    )
    validation_data = {
        "validation" : {
            "base_metrics": base_metrics,
            "lora_metrics": lora_metrics
        }
    }

    update_manifest(manifest_f_path, validation_data)

    Path(f"{flow_run_dir_path}/reports/").mkdir(parents=True, exist_ok=True)

# region make .md report
    md_report = generate_validation_report(
        manifest=manifest,
        metrics_base=base_metrics,
        metrics_lora=lora_metrics,
    )

    with open(os.path.join(f'{flow_run_dir_path}/reports/', 'report.md'), 'w', encoding="utf-8") as f:
        f.write(md_report)
# endregion

# region make metrics plots
    make_metrics_plot(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        metrics_model_base=base_metrics,
        metrics_model_lora=lora_metrics,
    )
# endregion

if __name__ == "__main__":
    COMMIT = "7c01ee7d6b644dbf4d5ccc2b9c1db9adab96b34a"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'

    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)
