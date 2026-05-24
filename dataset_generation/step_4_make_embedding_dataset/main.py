import json
import os
import random
import subprocess
from datetime import datetime
from common.constants import *
from typing import Dict, List, Tuple
from prefect import task

from common.constants import (
    DATA_DIR_NAME,
    GEN_NPC_ANSWER_DIR_NAME,
    EMBEDDING_DATASET_DIR_NAME,
)
from common.helpers import (
    format_action_signature,
    list_files,
    load_jsonl_to_dict,
    save_dict_records_to_jsonl,
    update_manifest,
)

black_list_for_dialogs_per_action = os.getenv('STEP_4E_BLACK_LIST_FOR_DIALOGS_PER_ACTION', '').split(',')


def _record_key(npc_response: dict) -> Tuple[str, str, str, dict]:
    """Return (action_name, params_key, full_key, params_dict) for a step_3 record."""
    action = npc_response.get('action') or {}
    action_name = action.get('name', 'unknown')
    params_dict = action.get('parameters', {}) or {}
    params_key = json.dumps(params_dict, sort_keys=True)
    full_key = f'{action_name}|{params_key}'
    return action_name, params_key, full_key, params_dict


def _signature_for_record(r: dict) -> str:
    return format_action_signature(r['_action_name'], r['_params'])


def _generate_pairs_user_request_mode(
        records: List[dict],
        pairs_per_level: int,
        score_high: float,
        score_mid: float,
        score_low: float,
        rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    """Original mode: sentence2 is another user_request.

    Suffers from label noise inherited from step_3 (semantically similar
    requests can be labeled with different actions). Kept for back-compat.
    """
    by_full_key: Dict[str, List[dict]] = {}
    by_action_name: Dict[str, List[dict]] = {}
    for r in records:
        by_full_key.setdefault(r['_full_key'], []).append(r)
        by_action_name.setdefault(r['_action_name'], []).append(r)
    all_action_names = list(by_action_name.keys())

    seen: Dict[frozenset, float] = {}
    pairs: List[dict] = []
    stats = {'high': 0, 'mid': 0, 'low': 0}

    def _emit(anchor: dict, other: dict, score: float, level: str):
        text_a = anchor['_text']
        text_b = other['_text']
        if text_a == text_b:
            return
        key = frozenset({text_a, text_b})
        if key in seen:
            return
        seen[key] = score
        pairs.append({
            'sentence1': text_a,
            'sentence2': text_b,
            'score': score,
            'anchor_action': anchor['_action_name'],
        })
        stats[level] += 1

    for anchor in records:
        same_full = [r for r in by_full_key[anchor['_full_key']] if r is not anchor]
        if same_full:
            picks = rng.sample(same_full, min(pairs_per_level, len(same_full)))
            for p in picks:
                _emit(anchor, p, score_high, 'high')

        same_action = [
            r for r in by_action_name[anchor['_action_name']]
            if r['_full_key'] != anchor['_full_key']
        ]
        if same_action:
            picks = rng.sample(same_action, min(pairs_per_level, len(same_action)))
            for p in picks:
                _emit(anchor, p, score_mid, 'mid')

        other_actions = [a for a in all_action_names if a != anchor['_action_name']]
        if other_actions:
            n_pick = min(pairs_per_level, len(other_actions))
            picked_actions = rng.sample(other_actions, n_pick)
            for act in picked_actions:
                pool = by_action_name[act]
                if pool:
                    _emit(anchor, rng.choice(pool), score_low, 'low')

    return pairs, stats


def _generate_pairs_action_signature_mode(
        records: List[dict],
        pairs_per_level: int,
        score_high: float,
        score_mid: float,
        score_low: float,
        rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    """Retrieval mode: sentence2 is the canonical action signature string.

    For each anchor (user_request) we emit:
      - 1 high pair: (anchor_text, anchor_signature, score_high)
      - up to N mid pairs: (anchor_text, same_action_other_params_signature, score_mid)
      - up to N low pairs: (anchor_text, other_action_signature, score_low)
    where N = pairs_per_level. Signatures are deterministic strings derived from
    action.name + sorted parameters, so there is no label noise from step_3.
    """
    # Index all unique full_keys → signature + one prototype record for params
    full_key_to_signature: Dict[str, str] = {}
    full_keys_by_action: Dict[str, List[str]] = {}
    for r in records:
        if r['_full_key'] not in full_key_to_signature:
            full_key_to_signature[r['_full_key']] = _signature_for_record(r)
            full_keys_by_action.setdefault(r['_action_name'], []).append(r['_full_key'])
    all_action_names = list(full_keys_by_action.keys())

    # Dedup: (sentence1, sentence2, score) — same anchor text shouldn't emit
    # the same pair more than once. Across anchors, the same canonical pair
    # can appear (e.g. two players asking the same thing); we keep only one.
    seen: set = set()
    pairs: List[dict] = []
    stats = {'high': 0, 'mid': 0, 'low': 0}

    def _emit(anchor_text: str, signature: str, score: float, level: str,
              anchor_action: str):
        key = (anchor_text, signature, score)
        if key in seen:
            return
        seen.add(key)
        pairs.append({
            'sentence1': anchor_text,
            'sentence2': signature,
            'score': score,
            'anchor_action': anchor_action,
        })
        stats[level] += 1

    for anchor in records:
        anchor_text = anchor['_text']
        anchor_action = anchor['_action_name']
        anchor_full_key = anchor['_full_key']

        # --- High: canonical signature for this anchor's action+params ---
        _emit(anchor_text, full_key_to_signature[anchor_full_key],
              score_high, 'high', anchor_action)

        # --- Mid: signatures of other variants of the same action ---
        sibling_keys = [
            k for k in full_keys_by_action.get(anchor_action, [])
            if k != anchor_full_key
        ]
        if sibling_keys:
            picks = rng.sample(sibling_keys, min(pairs_per_level, len(sibling_keys)))
            for k in picks:
                _emit(anchor_text, full_key_to_signature[k],
                      score_mid, 'mid', anchor_action)

        # --- Low: signatures of other actions (random variant per action) ---
        other_actions = [a for a in all_action_names if a != anchor_action]
        if other_actions:
            n_pick = min(pairs_per_level, len(other_actions))
            picked_actions = rng.sample(other_actions, n_pick)
            for act in picked_actions:
                variant_keys = full_keys_by_action.get(act, [])
                if variant_keys:
                    chosen_key = rng.choice(variant_keys)
                    _emit(anchor_text, full_key_to_signature[chosen_key],
                          score_low, 'low', anchor_action)

    return pairs, stats


def _generate_pairs_for_partition(
        records: List[dict],
        mode: str,
        pairs_per_level: int,
        score_high: float,
        score_mid: float,
        score_low: float,
        rng: random.Random,
) -> Tuple[List[dict], Dict[str, int]]:
    if mode == SENTENCE2_MODE_ACTION_SIGNATURE:
        return _generate_pairs_action_signature_mode(
            records, pairs_per_level, score_high, score_mid, score_low, rng
        )
    elif mode == SENTENCE2_MODE_USER_REQUEST:
        return _generate_pairs_user_request_mode(
            records, pairs_per_level, score_high, score_mid, score_low, rng
        )
    else:
        raise ValueError(
            f"Unknown STEP_4E_SENTENCE2_MODE={mode!r}. "
            f"Allowed: {SENTENCE2_MODE_USER_REQUEST}, {SENTENCE2_MODE_ACTION_SIGNATURE}"
        )


@task(name="step_4_make_embedding_dataset")
def process(git_commit: str, npc_name: str, flow_run_id: str, sentence2_mode: str = SENTENCE2_MODE_USER_REQUEST):
    val_ratio = float(os.getenv('STEP_4E_VAL_RATIO', 0.1))
    pairs_per_level = int(os.getenv('STEP_4E_PAIRS_PER_LEVEL', 2))
    score_high = float(os.getenv('STEP_4E_SCORE_HIGH', 1.0))
    score_mid = float(os.getenv('STEP_4E_SCORE_MID', 0.5))
    score_low = float(os.getenv('STEP_4E_SCORE_LOW', 0.0))
    seed = int(os.getenv('STEP_4E_SEED', 42))

    if sentence2_mode not in (SENTENCE2_MODE_USER_REQUEST,
                              SENTENCE2_MODE_ACTION_SIGNATURE):
        raise ValueError(
            f"Invalid STEP_4E_SENTENCE2_MODE={sentence2_mode!r}. "
            f"Allowed: {SENTENCE2_MODE_USER_REQUEST}, "
            f"{SENTENCE2_MODE_ACTION_SIGNATURE}"
        )
    print(f'sentence2 mode: {sentence2_mode}')

    rng = random.Random(seed)

    dialogs_glob = (
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/'
        f'{GEN_NPC_ANSWER_DIR_NAME}/*.jsonl'
    )
    dialog_files = list_files(dialogs_glob)

    # ---------- Phase 1: load & index ----------
    records: List[dict] = []
    for f_path in dialog_files:
        is_ok = True
        for prohibited in black_list_for_dialogs_per_action:
            if prohibited and prohibited in f_path:
                print(f'==> skipping {f_path}')
                is_ok = False
                break
        if not is_ok:
            continue

        dialogs = load_jsonl_to_dict(f_path)
        for d in dialogs:
            text = (d.get('usr_request') or {}).get('request', '')
            if not text:
                continue
            action_name, _params_key, full_key, params_dict = _record_key(
                d.get('npc_response') or {}
            )
            records.append({
                '_text': text,
                '_action_name': action_name,
                '_full_key': full_key,
                '_params': params_dict,
            })

    if not records:
        print('No records found — nothing to do.')
        return

    # ---------- Phase 2: stratified split records -> train / val ----------
    by_full_key_all: Dict[str, List[dict]] = {}
    variants_by_action: Dict[str, List[str]] = {}
    for r in records:
        if r['_full_key'] not in by_full_key_all:
            by_full_key_all[r['_full_key']] = []
            variants_by_action.setdefault(r['_action_name'], []).append(r['_full_key'])
        by_full_key_all[r['_full_key']].append(r)

    train_records: List[dict] = []
    val_records: List[dict] = []

    for action_name, full_keys in variants_by_action.items():
        min_variant_size = min(len(by_full_key_all[k]) for k in full_keys)
        n_val_per_variant = max(1, int(min_variant_size * val_ratio))

        for full_key in full_keys:
            items = list(by_full_key_all[full_key])
            rng.shuffle(items)

            val_records.extend(items[:n_val_per_variant])
            train_records.extend(items[n_val_per_variant:])

            params = full_key.split('|', 1)[1]
            print(f'  val {action_name} [{params}]: {n_val_per_variant} '
                  f'(train: {max(0, len(items) - n_val_per_variant)})')

    # Sanity: no text leakage between train and val
    train_texts = {r['_text'] for r in train_records}
    val_texts = {r['_text'] for r in val_records}
    leak = train_texts & val_texts
    if leak:
        print(f'WARNING: {len(leak)} text(s) leak between train and val '
              f'(same usr_request used for multiple action labels)')

    # ---------- Phase 3: generate pairs per partition ----------
    train_pairs, train_stats = _generate_pairs_for_partition(
        train_records, sentence2_mode, pairs_per_level,
        score_high, score_mid, score_low,
        rng,
    )
    val_pairs, val_stats = _generate_pairs_for_partition(
        val_records, sentence2_mode, pairs_per_level,
        score_high, score_mid, score_low,
        rng,
    )

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    # ---------- Phase 4: save (one file per anchor_action) ----------
    base_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{EMBEDDING_DATASET_DIR_NAME}/{sentence2_mode}'

    def _group_by_anchor_action(pairs: List[dict]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for p in pairs:
            out.setdefault(p['anchor_action'], []).append(p)
        return out

    train_by_action = _group_by_anchor_action(train_pairs)
    val_by_action = _group_by_anchor_action(val_pairs)

    train_pairs_per_action: Dict[str, int] = {}
    val_pairs_per_action: Dict[str, int] = {}

    for action_name, group in train_by_action.items():
        save_dict_records_to_jsonl(
            records=group,
            output_file=f'{action_name}.jsonl',
            folder_path=f'{base_dir}/training',
            append=False,
        )
        train_pairs_per_action[action_name] = len(group)

    for action_name, group in val_by_action.items():
        save_dict_records_to_jsonl(
            records=group,
            output_file=f'{action_name}.jsonl',
            folder_path=f'{base_dir}/validation',
            append=False,
        )
        val_pairs_per_action[action_name] = len(group)

    print(f'\nTrain: {len(train_pairs)} pairs '
          f'(high={train_stats["high"]}, mid={train_stats["mid"]}, low={train_stats["low"]}) '
          f'across {len(train_pairs_per_action)} action(s)')
    print(f'Val:   {len(val_pairs)} pairs '
          f'(high={val_stats["high"]}, mid={val_stats["mid"]}, low={val_stats["low"]}) '
          f'across {len(val_pairs_per_action)} action(s)')

    # ---------- Phase 5: manifest ----------
    pipeline_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

    manifest = {
        'type': 'embedding',
        'initial_args': {
            'unreal_commit': git_commit,
            'npc_name': npc_name,
            'pipeline_commit': pipeline_commit[:7],
            'timestamp': datetime.now().isoformat(),
            'flow_run_id': flow_run_id,
        },
        'dataset': {
            'dir_path': f'{base_dir}',
            'training': {
                'pairs_total': len(train_pairs),
                'by_score': {
                    str(score_high): train_stats['high'],
                    str(score_mid): train_stats['mid'],
                    str(score_low): train_stats['low'],
                },
                'actions': train_pairs_per_action,
                'anchors_total': len(train_records),
            },
            'validation': {
                'pairs_total': len(val_pairs),
                'by_score': {
                    str(score_high): val_stats['high'],
                    str(score_mid): val_stats['mid'],
                    str(score_low): val_stats['low'],
                },
                'actions': val_pairs_per_action,
                'anchors_total': len(val_records),
            },
            'params': {
                'val_ratio': val_ratio,
                'pairs_per_level': pairs_per_level,
                'seed': seed,
                'sentence2_mode': sentence2_mode,
            },
        },
    }

    manifest_f_name = f'{base_dir}/manifest.json'
    update_manifest(manifest_f_name, manifest)


if __name__ == '__main__':
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID, sentence2_mode=SENTENCE2_MODE_USER_REQUEST)
    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID, sentence2_mode=SENTENCE2_MODE_ACTION_SIGNATURE)
