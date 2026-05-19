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

def process(git_commit: str, npc_name: str, flow_run_id: str, dataset_name: str):
    black_list = [
        'NotEnoughGoldToBuy',
        'OutOfStock',
        'DoNothing'
    ]
    threshold = 0.1
    TOP_K = 3
    flow_run_dir_path = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}'

    manifest_f_path = os.path.abspath(f'{flow_run_dir_path}/manifest.json')

    with open(manifest_f_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        model_name = manifest["lora_training"]["model_name"].lower()
        llm_model_f_path = f'{DATA_DIR_NAME}/models/{model_name}_q4_k_m.gguf'
        #llm_model_f_path = f'input_data/models/qwen3.5-4b_q4_k_m.gguf'
        lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_{dataset_name}_lora_f16.gguf'
        #lora_adapter_f_path = f'{flow_run_dir_path}/{GGUF_DIR_NAME}/{model_name}_lora_f16.gguf'

    ullama_inference_cfg = make_ullama_config(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        model=llm_model_f_path,
        lora=lora_adapter_f_path,
        sp='tool_calling_system_prompt.txt'
    )
    ullama_inference_cfg['temp'] = 0.0
    #del ullama_inference_cfg['lora_adapter']
    del ullama_inference_cfg['grammar']

    ullama = ULlamaHelper(ullama_inference_cfg)

    emb_model_f_path = os.getenv('EMB_MODEL_F_PATH', 'input_data/models/baai_bge-base-en-v1.5_f16.gguf')
    lora_emb_model_f_path = os.getenv('LORA_EMB_MODEL_F_PATH', '')
    ENCODING = "utf-8"

    api = ULlamaWrapper()

    kb_f_path = f"{flow_run_dir_path}/knowledge_base.json"
    with open(kb_f_path, "r", encoding="utf-8") as f:
        kb_lst = json.loads(f.read())
        #kb_lst = kb_lst[0::3]
        emb_model = api.lib.ullama_load_model(emb_model_f_path.encode(ENCODING))
        kb_worker_ptr = api.lib.ullama_kb_make()

        kb_cfg = {
            "model": emb_model_f_path,
            "n_gpu_layers": 0,
            "lora_adapter": lora_emb_model_f_path,
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            #"document_prefix": "",

            #"reranker_model": "D:/Projects/Python/ai_npc_lora_dataset_gen/input_data/models/bge-reranker-v2-m3-Q8_0.gguf",
            #"reranker_n_ctx": 512,
            #"reranker_n_gpu_layers": 10,
            #"k_retrieve": 30,

            #"hybrid_search": False,
            #"bm25_stopwords": []
        }

        kb_cgf_str = json.dumps(kb_cfg).encode(ENCODING)
        kb_init_result = api.lib.ullama_kb_init(kb_worker_ptr, kb_cgf_str, emb_model)

        use_llm = False

        if kb_init_result:
            for kb_record in kb_lst:
                request = kb_record["request"]
                api.lib.ullama_kb_add_chunk(kb_worker_ptr, request.encode(ENCODING))
            api.lib.ullama_kb_update(kb_worker_ptr)

            validation_dataset_dir_path = f'{flow_run_dir_path}/{DATASET_DIR_NAME}/{dataset_name}_validation_custom/*.jsonl'

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
                    if use_llm:
                        llm_found_action, think_block = ullama.chat(
                            model=ullama_inference_cfg.get('model'),
                            system_prompt=ullama_inference_cfg.get('system_prompt', ''),
                            user_prompt=request
                        )

                    request_obj = json.loads(request)
                    if use_llm:
                        if valid_response_dict != llm_found_action:
                            llm_fails[file_name] += 1
                            print(f'   LLM Error: {request_obj["request"]}')
                            print(f'       valid: {valid_response_dict["action"]} != found: {llm_found_action["action"]}')

                    #request_obj['request'] = "Represent this sentence for searching relevant passages: " + request_obj['request']
                    request = json.dumps(request_obj)
                    results = api.search_top_n(
                        kb_handle=kb_worker_ptr,
                        query=request,
                        top_k=TOP_K
                    )
                    chunk_found = len(results)
                    if chunk_found:
                        found = False
                        found_actions = []
                        for result in results:
                            index = result[0]
                            score = result[1]
                            if score > threshold:
                                chunk = kb_lst[index]
                                found_action = chunk["action"]
                                if found_action not in found_actions:
                                    found_actions.append(found_action)
                                if found_action == target_action:
                                    found = True
                                    break
                                continue
                                if found_action != target_action:
                                    emb_fails[file_name] += 1
                                    print(f'   EMB Error: {request_obj["request"]}')
                                    print(f'       valid: {target_action} != found: {found_action}')

                        if not found:
                            emb_fails[file_name] += 1
                            print(f'   EMB Error: {request_obj["request"]}')
                            print(f'       valid: {target_action} != found: {found_actions}')
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
    DATASET_NAME = os.getenv("DATASET_NAME", 'chat')
    exit(
        process(
            git_commit=COMMIT,
            npc_name=NPC_NAME,
            flow_run_id=FLOW_RUN_ID,
            dataset_name=DATASET_NAME
        )
    )