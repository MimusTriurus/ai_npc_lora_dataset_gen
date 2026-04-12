import ctypes
import json
import os
from pathlib import Path
from typing import Dict

from ullama_python.ullama import ULlamaWrapper
from common.constants import DATA_DIR_NAME, DATASET_DIR_NAME, GGUF_DIR_NAME
from common.helpers import read_file, list_files, read_dataset_file
from common.inference import make_ullama_config
from common.ullama_helper import ULlamaHelper

def process(git_commit: str, npc_name: str, flow_run_id: str):
    black_list = [
        'NotEnoughGoldToBuy',
        'OutOfStock',
    ]
    threshold = 0.7
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'

    manifest_f_path = os.path.abspath(f'{flow_run_dir_path}/manifest.json')

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'
    ullama_inference_cfg = make_ullama_config(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        model=llm_model_f_path,
        lora=lora_adapter_f_path,
        sp='tool_calling_system_prompt.txt'
    )
    #del ullama_inference_cfg['lora_adapter']
    del ullama_inference_cfg['grammar']

    ullama = ULlamaHelper(ullama_inference_cfg)

    emb_model_f_path = os.getenv('EMB_MODEL_F_PATH', 'input_data/models/retriever.gguf')
    ENCODING = "utf-8"

    api = ULlamaWrapper()

    kb_f_path = f"{flow_run_dir_path}/knowledge_base.json"
    with open(kb_f_path, "r", encoding="utf-8") as f:
        kb_lst = json.loads(f.read())
        emb_model = api.lib.ullama_loadModel(emb_model_f_path.encode(ENCODING))
        kb_worker_ptr = api.lib.ullama_kb_make()
        kb_cfg = {
          "model": emb_model_f_path,
          "n_gpu_layers": 1
        }
        kb_cgf_str = json.dumps(kb_cfg).encode(ENCODING)
        kb_init_result = api.lib.ullama_kb_init(kb_worker_ptr, kb_cgf_str, emb_model)

        kb_chunk_idx = ctypes.c_int()
        kb_chunk_score = ctypes.c_float()

        if kb_init_result:
            for kb_record in kb_lst:
                request = kb_record["request"]
                api.lib.ullama_kb_addChunk(kb_worker_ptr, request.encode(ENCODING))
            api.lib.ullama_kb_update(kb_worker_ptr)

            validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/validation/*.jsonl'

            total_requests = 0

            llm_fails: Dict[str, int] = {}
            emb_fails: Dict[str, int] = {}

            dataset_files = list_files(validation_dataset_dir_path)
            for dataset_file in dataset_files:
                file_name = Path(dataset_file).stem

                if file_name in black_list:
                    continue

                llm_fails[file_name] = 0
                emb_fails[file_name] = 0
                counter = 1
                dataset_pairs = read_dataset_file(dataset_file)
                requests_count = len(dataset_pairs)
                for pair in dataset_pairs:
                    print(f'--> {file_name} [{counter}/{requests_count}]')
                    counter += 1
                    total_requests += 1
                    request = pair[0]
                    valid_response_dict = pair[1]
                    target_action = valid_response_dict['action']

                    llm_found_action, think_block = ullama.chat(
                        model=ullama_inference_cfg.get('model'),
                        system_prompt=ullama_inference_cfg.get('system_prompt', ''),
                        user_prompt=request
                    )

                    if target_action != llm_found_action:
                        llm_fails[file_name] += 1

                    request_obj = json.loads(request)
                    request_obj = {
                        "context": "",
                        "state_of_user": "User has 100 gold",
                        "request_of_user": request_obj['request']
                    }

                    request = json.dumps(request_obj).encode(ENCODING)

                    chunk_found = api.lib.ullama_kb_search(
                        kb_worker_ptr,
                        request,
                        ctypes.byref(kb_chunk_idx),
                        ctypes.byref(kb_chunk_score)
                    )
                    if chunk_found and kb_chunk_score.value > threshold:
                        chunk = kb_lst[kb_chunk_idx.value]
                        found_action = chunk["action"]
                        if found_action != target_action:
                            emb_fails[file_name] += 1
        else:
            print(f'Error on init knowledge base')
        print()
        print(f'--- LLM total fails: {sum(llm_fails.values())}/{total_requests} ---')
        print(f'--- Emb total fails: {sum(emb_fails.values())}/{total_requests} ---')
        print()
        print('llm_fails:')
        print(json.dumps(llm_fails, indent=4))
        print('emb_fails:')
        print(json.dumps(emb_fails, indent=4))

        print(f'=== End ===')

if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID))