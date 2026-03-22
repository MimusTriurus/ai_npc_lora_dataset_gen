import os
import os.path
import subprocess
import sys
from pathlib import Path
from common.constants import *
from prefect import task

@task(name="step_1_convert_to_gguf")
def process(
        git_commit: str,
        npc_name: str,
        flow_run_id: str,
        base_model: str,
):
    LORA_PATH = f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{LORA_DIR_NAME}"
    LORA_ADAPTER_PATH = f"{LORA_PATH}/final_adapter/"

    BASE_MODEL = base_model
    OUT_FORMAT = os.getenv('STEP_1_OUT_FORMAT', 'f16')

    OUT_BASE_MODEL_DIR = f"{DATA_DIR_NAME}/models/"
    OUT_BASE_MODEL_FILE = Path(f"{OUT_BASE_MODEL_DIR}/{BASE_MODEL.lower()}_{OUT_FORMAT.lower()}.gguf")
    OUT_LORA_ADAPTER_FILE = Path(f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GGUF_DIR_NAME}/{BASE_MODEL.lower()}_lora_{OUT_FORMAT}.gguf")

    os.makedirs(os.path.dirname(OUT_BASE_MODEL_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LORA_ADAPTER_FILE), exist_ok=True)

    LLAMA_CPP_DIR = Path(os.getenv('STEP_1_LLAMA_CPP_DIR', 'llama.cpp/'))
    LLAMA_BIN_DIR = Path(os.getenv('STEP_1_LLAMA_BIN_DIR', 'llama.cpp/bin'))

    sys.path.append(str(LLAMA_CPP_DIR))
    converter_path = str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py")

    if os.path.isfile(converter_path):
        model_f = f'models/{BASE_MODEL}'
        if not os.path.isfile(OUT_BASE_MODEL_FILE):
            print(f"===> Converting models/{BASE_MODEL} to the .gguf format")
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
            '--base', f'models/{BASE_MODEL}'
        ], check=True)

        print(f"Base model: {OUT_BASE_MODEL_FILE}")
        print(f"LoRA adapter:  {OUT_LORA_ADAPTER_FILE}")
    else:
        print(f"===> Error: can't find converter: {converter_path}")

    print(f"===> Quantization q4_k_m for. {OUT_BASE_MODEL_FILE}")

    quatizator_path = str(LLAMA_BIN_DIR / "llama-quantize.exe")

    BASE_MODEL_Q4 = f'{OUT_BASE_MODEL_DIR}/{BASE_MODEL.lower()}_q4_k_m.gguf'
    if not os.path.isfile(BASE_MODEL_Q4):
        if os.path.isfile(quatizator_path):
            subprocess.run([
                quatizator_path,
                OUT_BASE_MODEL_FILE,
                BASE_MODEL_Q4,
                'q4_k_m'
            ], check=True)
        else:
            print(f"===> Error: can't find quantizator: {quatizator_path}")

    print('\n Ready!')

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'
    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)