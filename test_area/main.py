import json
import os

from common.constants import DATA_DIR_NAME
from common.manifest import Manifest
from common.ullama_helper import ULlamaHelper
from ullama_python.ullama_python.ullama import ULlamaWrapper

def process(llm_dir_path: str, emb_dir_path: str):
    TOP_K = int(os.getenv('STEP_2_TOP_K', 1))
    EMB_THRESHOLD = float(os.getenv('STEP_2_EMB_THRESHOLD', 0.6))
    print('=== Start ===')
    llm_manifest_f_path = os.path.abspath(f'{llm_dir_path}/manifest.json')
    llm_manifest = Manifest(llm_manifest_f_path)
    llm_inference_cfg = json.loads(open(f'{llm_dir_path}/inference_cfg.json', "r", encoding="utf-8").read())

    inference = ULlamaHelper(llm_inference_cfg)
    # region KB initialization
    kb_f_path = f"{emb_dir_path}/knowledge_base.json"
    emb_manifest_f_path = os.path.abspath(f'{emb_dir_path}/manifest.json')
    emb_manifest = Manifest(emb_manifest_f_path)
    kb_chunks = []
    kb_lst = []
    with open(kb_f_path, "r", encoding="utf-8") as f:
        kb_lst = json.loads(f.read())
        for kb_record in kb_lst:
            chunk_text = kb_record.get('signature')
            if chunk_text is None:
                # Backward-compat with the legacy KB schema {request, action}.
                chunk_text = kb_record.get('request', '')
            kb_chunks.append(chunk_text)

    emb_inference_cfg = json.loads(open(f'{emb_dir_path}/inference_cfg.json', "r", encoding="utf-8").read())
    inference.kb_init(emb_inference_cfg, kb_chunks)

    requests = [
        'What do you have for sale?',
        'What kind of weapons do you have?',
        'Show me the pistol.',
        'Sell me the pistol.',
        'Ok, do you have ammo for this pistol?',
        'Ok, I\'ll take pistol\'s ammo',
        'Give me one more pack of pistol\'s ammo',
        'Do you have medications?'
    ]

    for request in requests:
        request_obj = {
            'request': f"{emb_manifest.emb_request_prefix()}{request}"
        }
        found_chunks = inference.kb_search(json.dumps(request_obj), TOP_K)
        print(f'==> Request: {request}')
        for chunk in found_chunks:
            idx, score = chunk
            if score >= EMB_THRESHOLD:
                print(f'--> [{score}] {kb_lst[idx]["action"]}')
        print('---------')
        print('')

    # endregion
    print('=== End ===')

if __name__ == '__main__':
    unreal_hash = os.getenv('COMMIT')
    npc_name = os.getenv('NPC_NAME')
    flow_run_id = os.getenv('FLOW_RUN_ID')

    llm_model = os.getenv('STEP_0_MODEL_NAME')
    llm_hash = os.getenv('LLM_TRAINING_SESSION_HASH')
    llm_lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora/{llm_model}/chat/{llm_hash}'

    emb_model = os.getenv('STEP_0_EMB_MODEL_NAME')
    emb_hash = os.getenv('EMB_TRAINING_SESSION_HASH')
    emb_lora_path = f'{DATA_DIR_NAME}/{unreal_hash}/{npc_name}/{flow_run_id}/training/lora_embedding/{emb_model}/action_signature/{emb_hash}'

    process(
        llm_lora_path,
        emb_lora_path
    )