import json
import os
from pathlib import Path
from typing import Dict
from prefect import task

from common.metrics_plot_generation import make_emb_metrics_plot
from common.training_results_report_generation import generate_validation_report
from ullama_python.ullama_python.ullama import ULlamaWrapper
from common.constants import DATASET_DIR_NAME, CHAT_LLM_PREFIX, DATA_DIR_NAME
from common.helpers import list_files, read_dataset_file
from common.manifest import Manifest

QUERY_PREFIX="Represent this sentence for searching relevant passages: "

@task(name="step_2_lora_validation")
def process(model_dir_path: str, use_lora: bool = True):
    black_list = os.getenv('ACTIONS_BLACK_LIST', '').split(',')
    threshold = float(os.getenv('STEP_2_EMB_THRESHOLD', 0.4))
    TOP_K = int(os.getenv('STEP_2_TOP_K', 1))

    manifest_f_path = os.path.abspath(f'{model_dir_path}/manifest.json')
    manifest = Manifest(manifest_f_path)
    ullama_inference_cfg = json.loads(open(f'{model_dir_path}/inference_cfg.json', "r", encoding="utf-8").read())

    if not use_lora:
        del ullama_inference_cfg['lora_adapter']

    ENCODING = "utf-8"
    api = ULlamaWrapper()
    # ----- Build knowledge base of action signatures -----------------------
    kb_f_path = f"{model_dir_path}/knowledge_base.json"
    with open(kb_f_path, "r", encoding="utf-8") as f:
        kb_lst = json.loads(f.read())

    kb_worker_ptr = api.lib.ullama_kb_make()
    ullama_inference_cfg_str = json.dumps(ullama_inference_cfg).encode(ENCODING)
    emb_model = api.lib.ullama_load_model(ullama_inference_cfg_str)
    kb_init_result = api.lib.ullama_kb_init(kb_worker_ptr, ullama_inference_cfg_str, emb_model)

    if not kb_init_result:
        print('Error on init knowledge base')
        return

    # KB chunks are action signatures — same format used as sentence2 in
    # the embedding LoRA training (action_signature mode). This keeps the
    # retrieval task (user request -> action signature) identical to training.
    for kb_record in kb_lst:
        chunk_text = kb_record.get('signature')
        if chunk_text is None:
            # Backward-compat with the legacy KB schema {request, action}.
            chunk_text = kb_record.get('request', '')
        api.lib.ullama_kb_add_chunk(kb_worker_ptr, chunk_text.encode(ENCODING))
    api.lib.ullama_kb_update(kb_worker_ptr)

    validation_dataset_dir_path = (
        f'{manifest.flow_dir_path()}/{DATASET_DIR_NAME}/{CHAT_LLM_PREFIX}/'
        f'validation/*.jsonl'
    )

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

            request_obj = json.loads(request)
            request_obj['request'] = manifest.emb_request_prefix() + request_obj['request']

            request = json.dumps(request_obj)

            results = api.search_top_n(
                kb_handle=kb_worker_ptr,
                query=request,
                top_k=TOP_K,
            )

            if not results:
                emb_fails[file_name] += 1
                print(f'   EMB Error: {request_obj["request"]}')
                print(f'       valid: {target_action} != found: <no hits>')
                continue

            found = False
            found_actions = []
            score = 0.0
            for result in results:
                index = result[0]
                score = result[1]
                if score <= threshold:
                    continue
                chunk = kb_lst[index]
                found_action = chunk["action"]
                if found_action not in found_actions:
                    found_actions.append(found_action)
                if found_action == target_action:
                    found = True
                    break

            if not found:
                emb_fails[file_name] += 1
                print(f'   [{score}] EMB Error: {request_obj["request"]}')
                print(f'       valid: {target_action} != found: {found_actions}')

    validation_results = {
        "TOP_K": TOP_K,
        "total_requests": total_requests,
        "total_fails" : sum(emb_fails.values()),
        "fails_per_action": emb_fails,
    }

    manifest.set_validation_results(validation_results)
    manifest.update()

    print()
    print(f'--- Emb total fails: {sum(emb_fails.values())}/{total_requests} ---')
    print()
    print('emb_fails:')
    print(json.dumps(emb_fails, indent=4))
    print(f'=== End ===')

    return validation_results, manifest


if __name__ == "__main__":
    unreal_hash = os.getenv('COMMIT')
    npc_name = os.getenv('NPC_NAME')
    flow_run_id = os.getenv('FLOW_RUN_ID')
    model = os.getenv('STEP_0_EMB_MODEL_NAME')
    hash = os.getenv('EMB_TRAINING_SESSION_HASH')
    #lora_path = f'input_data/7c01ee7/trader/v2/training/lora_embedding/{model}/user_request/{hash}'
    #process(lora_path)

    lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora_embedding/{model}/action_signature/{hash}'
    lora_errors, lora_manifest = process(model_dir_path=lora_path, use_lora=True)

    base_errors, base_manifest = process(model_dir_path=lora_path, use_lora=False)

    make_emb_metrics_plot(
        base_errors,
        lora_errors,
        lora_path
    )

    md_report = generate_validation_report(
        lora_manifest.to_dict(),
        metrics_base=base_errors,
        metrics_lora=lora_errors,
    )

    with open(os.path.join(f'{lora_path}/reports/', 'report.md'), 'w', encoding="utf-8") as f:
        f.write(md_report)

