import json
import os
import os.path
import subprocess
import sys
from pathlib import Path
from typing import Optional
from common.constants import *
from prefect import task

from common.helpers import update_manifest

# [OBSOLETE]
def _find_peft_adapter_dir(root: str) -> Optional[str]:
    root_path = Path(root)
    if (root_path / "adapter_config.json").is_file():
        return str(root_path)
    for child in sorted(root_path.iterdir()):
        if child.is_dir() and (child / "adapter_config.json").is_file():
            return str(child)
    return None

def make_inference_config(model_dir_path: str):
    manifest_f_path = f'{model_dir_path}/manifest.json'
    manifest = json.loads(open(manifest_f_path, 'r').read())

    initial_args = manifest['initial_args']
    unreal_commit = initial_args['unreal_commit']
    npc_name = initial_args['npc_name']
    flow_run_id = initial_args['flow_run_id']
    root_dir =f'{DATA_DIR_NAME}/{unreal_commit}/{npc_name}/{flow_run_id}'
    model_type = manifest['type']
    inference_config_f_name = f'{root_dir}/{model_type}_inference_cfg.json'
    try:
        with open(inference_config_f_name, 'r') as f:
            inference_cfg = json.load(f)
            inference_cfg['model'] = f"{manifest['gguf']['model_f_path']}"
            inference_cfg['lora_adapter'] = f"{manifest['gguf']['lora_f_path']}"
        with open(f'{model_dir_path}/inference_cfg.json', 'w') as f:
            json.dump(inference_cfg, f, indent=4)
    except FileNotFoundError as e:
        print(f'Convert to gguf. make_inference_config. {e}')

@task(name="step_1_convert_to_gguf")
def process(
    lora_dir_path: str,
    quantization: bool = False,
):
    LORA_PATH = f"{lora_dir_path}"
    LORA_ADAPTER_PATH = f"{LORA_PATH}/final_adapter/"

    manifest_f_path = f'{LORA_PATH}/manifest.json'
    manifest = json.loads(open(manifest_f_path, 'r').read())
    model_name = manifest['training']['model_name']

    MODEL_SLUG = model_name.replace("/", "_")
    BASE_MODEL_HF_DIR = f"models/{MODEL_SLUG}"
    OUT_FORMAT = os.getenv('STEP_1_OUT_FORMAT', 'f16')

    OUT_BASE_MODEL_DIR = f"{DATA_DIR_NAME}/models/"
    OUT_BASE_MODEL_FILE = Path(f"{OUT_BASE_MODEL_DIR}/{MODEL_SLUG}.gguf")
    OUT_LORA_ADAPTER_FILE = Path(f"{lora_dir_path}/lora.gguf")

    os.makedirs(os.path.dirname(OUT_BASE_MODEL_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LORA_ADAPTER_FILE), exist_ok=True)

    LLAMA_CPP_DIR = Path(os.getenv('STEP_1_LLAMA_CPP_DIR', 'llama.cpp/'))
    LLAMA_BIN_DIR = Path(os.getenv('STEP_1_LLAMA_BIN_DIR', 'llama.cpp/bin'))

    sys.path.append(str(LLAMA_CPP_DIR))
    converter_path = str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py")

    if os.path.isfile(converter_path):
        model_f = BASE_MODEL_HF_DIR
        if not os.path.isfile(OUT_BASE_MODEL_FILE):
            print(f"===> Converting models/{MODEL_SLUG} to the .gguf format")
            subprocess.run([
                sys.executable, str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
                model_f,
                '--outfile', OUT_BASE_MODEL_FILE,
                '--outtype', OUT_FORMAT
            ], check=True)

        print(f"===> Converting a LoRA adapter to the .gguf format. {LORA_ADAPTER_PATH}")
        subprocess.run([
            sys.executable, str(LLAMA_CPP_DIR / "convert_lora_to_gguf.py"),
            LORA_ADAPTER_PATH,
            '--outfile', OUT_LORA_ADAPTER_FILE,
            '--outtype', OUT_FORMAT,
            '--base', model_f
        ], check=True)

        print(f"Base model: {OUT_BASE_MODEL_FILE}")
        print(f"LoRA adapter:  {OUT_LORA_ADAPTER_FILE}")
    else:
        print(f"===> Error: can't find converter: {converter_path}")

    if quantization:
        print(f"===> Quantization q4_k_m for. {OUT_BASE_MODEL_FILE}")
        quatizator_path = str(LLAMA_BIN_DIR / "llama-quantize.exe")
        BASE_MODEL_Q4 = Path(f'{OUT_BASE_MODEL_DIR}/{MODEL_SLUG}_q4_k_m.gguf')
        if not os.path.isfile(BASE_MODEL_Q4):
            if os.path.isfile(quatizator_path):
                subprocess.run([
                    quatizator_path,
                    OUT_BASE_MODEL_FILE,
                    str(BASE_MODEL_Q4.resolve()),
                    'q4_k_m'
                ], check=True)
            else:
                print(f"===> Error: can't find quantizator: {quatizator_path}")

    manifest['gguf'] = {
        'lora_f_path': str(OUT_LORA_ADAPTER_FILE.as_posix()),
        'model_f_path': str(OUT_BASE_MODEL_FILE.as_posix())
    }
    update_manifest(manifest_f_path, manifest)

    make_inference_config(lora_dir_path)
    print('\n Ready!')

if __name__ == '__main__':
    hash = os.getenv('TRAINING_SESSION_HASH')
    lora_path = f'input_data/7c01ee7/trader/v2/training/lora_embedding/BAAI/bge-base-en-v1.5/user_request/{hash}'
    process(lora_path)
    lora_path = f'input_data/7c01ee7/trader/v2/training/lora_embedding/BAAI/bge-base-en-v1.5/action_signature/{hash}'
    process(lora_path)