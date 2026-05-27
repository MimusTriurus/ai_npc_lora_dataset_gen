import json
import warnings
import os
import logging

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from prefect import task
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import (
    CoSENTLoss,
    MultipleNegativesRankingLoss,
)
from transformers import EarlyStoppingCallback

LOSS_MNR = 'mnr'
LOSS_COSENT = 'cosent'

# Recommended BGE query-side prefix (used at pretraining time for retrieval).
# Reference: https://huggingface.co/BAAI/bge-base-en-v1.5
BGE_QUERY_PREFIX_DEFAULT = ''  # set empty to keep prefix opt-in

from common.constants import (
    DATA_DIR_NAME,
    EMBEDDING_DATASET_DIR_NAME,
    LORA_EMBEDDING_DIR_NAME,
    MODELS_DIR_NAME, SENTENCE2_MODE_USER_REQUEST, SENTENCE2_MODE_ACTION_SIGNATURE,
)
from common.helpers import update_manifest, make_hash

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


def _load_cosent_dataset(folder_glob: str, query_prefix: str):
    """Load *.jsonl pairs in CoSENT format: (sentence1, sentence2, label).

    If `query_prefix` is non-empty, it is prepended to sentence1 only
    (asymmetric query→passage retrieval setup).
    """
    ds = load_dataset("json", data_files=folder_glob, split="train")

    keep = {"sentence1", "sentence2", "score"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    if query_prefix:
        ds = ds.map(lambda r: {"sentence1": f"{query_prefix}{r['sentence1']}"})

    # CoSENTLoss expects float labels in `label` column
    ds = ds.rename_column("score", "label")
    return ds


def _load_mnr_dataset(
        folder_glob: str,
        query_prefix: str,
        score_high: float,
        score_mid: float,
        score_low: float,
):
    """Reshape pairs into (anchor, positive, negative) triplets for MNR.

    Logic per unique anchor (sentence1):
      - positive = sentence2 from a high-score pair (must exist — else skip anchor)
      - negative = sentence2 from a mid-score pair (preferred, harder negative)
                   or a low-score pair (fallback)

    If no anchor has a negative available, fall back to (anchor, positive) only —
    MNR will use in-batch negatives. The two schemas don't mix in one Dataset,
    so anchors lacking a hard negative are dropped when others have one.
    """
    raw = load_dataset("json", data_files=folder_glob, split="train")

    by_anchor: dict = {}
    for row in raw:
        s1 = row.get('sentence1')
        s2 = row.get('sentence2')
        score = row.get('score')
        if s1 is None or s2 is None:
            continue
        bucket = by_anchor.setdefault(s1, {'high': [], 'mid': [], 'low': []})
        if score == score_high:
            bucket['high'].append(s2)
        elif score == score_mid:
            bucket['mid'].append(s2)
        elif score == score_low:
            bucket['low'].append(s2)

    triplets = []
    pairs_only = []
    for s1, b in by_anchor.items():
        if not b['high']:
            continue
        anchor = f"{query_prefix}{s1}" if query_prefix else s1
        positive = b['high'][0]
        if b['mid']:
            triplets.append({
                'anchor': anchor, 'positive': positive, 'negative': b['mid'][0]
            })
        elif b['low']:
            triplets.append({
                'anchor': anchor, 'positive': positive, 'negative': b['low'][0]
            })
        else:
            pairs_only.append({'anchor': anchor, 'positive': positive})

    if triplets and pairs_only:
        logger.warning(
            f"Dropping {len(pairs_only)} anchors with no hard negative "
            f"(mixing triplet/pair schema is not supported in one Dataset)"
        )
        return Dataset.from_list(triplets)
    if triplets:
        return Dataset.from_list(triplets)
    return Dataset.from_list(pairs_only)


# Known LoRA-trainable module suffixes across encoder/decoder architectures.
# Auto-detection (see _detect_target_modules) intersects these with what the
# loaded model actually exposes — so a new architecture only requires adding
# its suffixes here, no per-model branching in the training code.
_KNOWN_LORA_SUFFIXES = frozenset({
    # BERT / BGE / RoBERTa / DeBERTa
    'query', 'key', 'value', 'dense',
    # LLaMA / Qwen / Mistral / Gemma / Phi3 (decoder-style attn + FFN)
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
    # Fallback for some HF wrappers
    'in_proj', 'out_proj',
})

# Module-name fragments to skip even when the suffix matches. Currently only
# the BERT pooler — llama.cpp's BERT GGUF converter has no mapping for
# pooler.dense.* and crashes on export. The pooler is unused by BGE at
# inference (mean-pooling is used), so excluding it is lossless.
_LORA_EXCLUDE_FRAGMENTS = ('pooler',)


def _detect_target_modules(model) -> str:
    """Build a PEFT-compatible regex for `target_modules` by introspecting
    the loaded model.

    Returns a regex string (not a list) so PEFT uses `re.fullmatch` on each
    module name. The pattern matches modules whose final name component is a
    known LoRA target suffix AND whose full path does not contain any of
    `_LORA_EXCLUDE_FRAGMENTS`.

    Examples produced:
        BGE/BERT:  ^(?!.*pooler).*\\.(dense|key|query|value)$
        Qwen3:     .*\\.(down_proj|gate_proj|k_proj|o_proj|q_proj|up_proj|v_proj)$
    """
    found = set()
    for name, _ in model.named_modules():
        suffix = name.rsplit('.', 1)[-1]
        if suffix not in _KNOWN_LORA_SUFFIXES:
            continue
        if any(frag in name for frag in _LORA_EXCLUDE_FRAGMENTS):
            continue
        found.add(suffix)

    if not found:
        raise RuntimeError(
            "Could not auto-detect LoRA target modules for this base model. "
            "Inspect `[name for name, _ in model.named_modules()]` and add "
            "the relevant attention/FFN suffixes to _KNOWN_LORA_SUFFIXES."
        )

    alt = '|'.join(sorted(found))
    # Only emit the negative-lookahead clause when we actually have anything
    # to exclude — keeps the pattern tidy on Qwen/LLaMA which have no pooler.
    if any(_check_any_module_contains(model, frag) for frag in _LORA_EXCLUDE_FRAGMENTS):
        return rf"^(?!.*({'|'.join(_LORA_EXCLUDE_FRAGMENTS)})).*\.({alt})$"
    return rf".*\.({alt})$"


def _check_any_module_contains(model, fragment: str) -> bool:
    return any(fragment in name for name, _ in model.named_modules())


@task(name="step_0_train_embedding_lora_adapter")
def process(
        git_commit: str,
        npc_name: str,
        flow_run_id: str,
        dataset_name: str = SENTENCE2_MODE_USER_REQUEST,
        base_model: str = "BAAI/bge-base-en-v1.5",
        # hyper parameters
        num_train_epoch: int = 1,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        batch_size: int = 64,
        loss_name: str = LOSS_MNR
):
    # Optional query-side prefix (BGE retrieval convention). Empty disables it.
    query_prefix = os.getenv('STEP_0E_QUERY_PREFIX', BGE_QUERY_PREFIX_DEFAULT)

    hyper_params_hash = make_hash(
        num_train_epoch,
        lora_rank,
        lora_alpha,
        batch_size,
        loss_name,
        query_prefix
    )[:7]
    output_dir = (
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/'
        f'{LORA_EMBEDDING_DIR_NAME}/{base_model}/{dataset_name}/{hyper_params_hash}'
    )

    if os.path.isdir(output_dir):
        print(f"{output_dir} already exists, skipping.")
        return

    """Train a LoRA adapter for an embedding model on the 3-level similarity
    pairs produced by `dataset_generation/step_4_make_embedding_dataset`.

    Dataset format (per *.jsonl in 3_embedding_dataset/embedding_{training,validation}):
        {"sentence1": "...", "sentence2": "...", "score": 1.0|0.5|0.0, "anchor_action": "..."}
    Loss: CoSENTLoss (handles the continuous 3-level scale natively).
    """
    model_path = f'{MODELS_DIR_NAME}/{base_model.replace("/", "_")}'

    if not os.path.exists(model_path):
        logger.info(f"Model not found locally, downloading from HuggingFace: {base_model}")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=base_model, local_dir=model_path)

    learning_rate = float(os.getenv('STEP_0E_LEARNING_RATE', 2e-5))
    gradient_accumulation = int(os.getenv('STEP_0E_GRADIENT_ACCUMULATION', 1))
    weight_decay = float(os.getenv('STEP_0E_WEIGHT_DECAY', 0.01))
    eval_strategy = os.getenv('STEP_0E_EVAL_STRATEGY', 'epoch')
    eval_steps = int(os.getenv('STEP_0E_EVAL_STEPS', 100))
    warmup_ratio = float(os.getenv('STEP_0E_WARMUP_RATIO', 0.1))
    seed = int(os.getenv('STEP_0E_SEED', 42))

    # Loss choice: MNR (default, retrieval-style) with CoSENT as fallback.

    if loss_name not in (LOSS_MNR, LOSS_COSENT):
        raise ValueError(
            f"Invalid STEP_0E_LOSS={loss_name!r}. Allowed: {LOSS_MNR}, {LOSS_COSENT}"
        )

    # Score thresholds — must match what step_4_make_embedding_dataset wrote.
    score_high = float(os.getenv('STEP_0E_SCORE_HIGH', 1.0))
    score_mid = float(os.getenv('STEP_0E_SCORE_MID', 0.5))
    score_low = float(os.getenv('STEP_0E_SCORE_LOW', 0.0))

    dataset_dir = (
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/'
        f'{EMBEDDING_DATASET_DIR_NAME}/{dataset_name}'
    )

    logger.info("=" * 50)
    logger.info(f"EMBEDDING LoRA TRAINING")
    logger.info("=" * 50)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Epochs: {num_train_epoch}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size (per device): {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation}")
    logger.info(f"LoRA rank/alpha: {lora_rank}/{lora_alpha}")
    logger.info(f"Loss: {loss_name}")
    logger.info(f"Query prefix: {query_prefix!r}" if query_prefix else "Query prefix: <none>")
    logger.info("=" * 50 + "\n")

    train_glob = f'{dataset_dir}/training/*.jsonl'
    val_glob = f'{dataset_dir}/validation/*.jsonl'

    if loss_name == LOSS_MNR:
        train_ds = _load_mnr_dataset(
            train_glob, query_prefix, score_high, score_mid, score_low
        )
        val_ds = _load_mnr_dataset(
            val_glob, query_prefix, score_high, score_mid, score_low
        )
        dataset_kind = (
            'triplets' if 'negative' in train_ds.column_names else 'pairs-in-batch'
        )
    else:  # LOSS_COSENT
        train_ds = _load_cosent_dataset(train_glob, query_prefix)
        val_ds = _load_cosent_dataset(val_glob, query_prefix)
        dataset_kind = 'cosent-pairs'

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Empty training set after loading ({loss_name} mode). "
            f"Run step_4_make_embedding_dataset first."
        )

    logger.info(
        f"Train rows: {len(train_ds)} | Val rows: {len(val_ds)} "
        f"({dataset_kind}, columns: {list(train_ds.column_names)})"
    )

    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_path)

    logger.info("Applying LoRA configuration...")
    # Auto-detect target modules — handles BERT-family (BGE) and decoder-style
    # backbones (Qwen3, LLaMA, Mistral, ...) without per-arch branching.
    # Excludes BERT pooler so the LoRA adapter can be exported to GGUF.
    target_modules_pattern = _detect_target_modules(model)
    logger.info(f"LoRA target_modules pattern: {target_modules_pattern}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules_pattern,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    # sentence-transformers >=3.0 routes add_adapter to the underlying transformer
    model.add_adapter(lora_config)

    # Sanity check: nothing in _LORA_EXCLUDE_FRAGMENTS should have been wrapped.
    leaked = [
        n for n, _ in model.named_parameters()
        if any(frag in n for frag in _LORA_EXCLUDE_FRAGMENTS)
        and ('lora_a' in n.lower() or 'lora_b' in n.lower())
    ]
    if leaked:
        raise RuntimeError(
            f"Excluded modules were unexpectedly wrapped by LoRA: {leaked}. "
            f"GGUF export will likely fail."
        )

    # Print trainable params manually (ST doesn't expose print_trainable_parameters)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    if loss_name == LOSS_MNR:
        loss = MultipleNegativesRankingLoss(model)
    else:
        loss = CoSENTLoss(model)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),

        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=eval_strategy,
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=10,
        logging_first_step=True,
        seed=seed,
        dataloader_drop_last=True,
        # report_to="tensorboard",
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )

    logger.info("Initializing trainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        callbacks=[early_stopping],
    )

    logger.info("Starting training...")
    trainer.train()

    save_dir = f'{output_dir}/final_adapter'
    logger.info(f"Saving SentenceTransformer model (with adapter) to: {save_dir}")
    model.save_pretrained(save_dir)

    logger.info("Releasing GPU memory...")
    del trainer
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("\nEmbedding LoRA training completed successfully!")

    manifest_f_name = (
        f'{dataset_dir}/manifest.json'
    )
    manifest = {
        'training': {
            'model_name': base_model,
            'model_path': model_path,
            'num_train_epoch': num_train_epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation': gradient_accumulation,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'seed': seed,
            'loss': (
                'MultipleNegativesRankingLoss' if loss_name == LOSS_MNR
                else 'CoSENTLoss'
            ),
            'dataset_kind': dataset_kind,
            'query_prefix': query_prefix,
            'pairs_train': len(train_ds),
            'pairs_val': len(val_ds),
        }
    }
    dataset_manifest = json.loads(open(manifest_f_name, 'r').read())
    manifest = manifest | dataset_manifest
    update_manifest(f'{output_dir}/manifest.json', manifest)

    return output_dir

if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")

    NUM_TRAIN_EPOCH = int(os.getenv('STEP_0_NUM_TRAIN_EPOCH', 1))
    if False:
        LOSS_NAME = LOSS_MNR
        process(
            git_commit=COMMIT,
            npc_name=NPC_NAME,
            flow_run_id=FLOW_RUN_ID,
            dataset_name=SENTENCE2_MODE_USER_REQUEST,
            loss_name=LOSS_NAME,
            num_train_epoch=NUM_TRAIN_EPOCH,
        )

        process(
            git_commit=COMMIT,
            npc_name=NPC_NAME,
            flow_run_id=FLOW_RUN_ID,
            dataset_name=SENTENCE2_MODE_ACTION_SIGNATURE,
            loss_name=LOSS_NAME,
            num_train_epoch=NUM_TRAIN_EPOCH,
        )

    LOSS_NAME = LOSS_COSENT
    model = os.getenv('STEP_0_EMB_MODEL_NAME')

    process(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id=FLOW_RUN_ID,
        dataset_name=SENTENCE2_MODE_USER_REQUEST,
        loss_name=LOSS_NAME,
        num_train_epoch=NUM_TRAIN_EPOCH,
        base_model=model,
        batch_size=8
    )

    process(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id=FLOW_RUN_ID,
        dataset_name=SENTENCE2_MODE_ACTION_SIGNATURE,
        loss_name=LOSS_NAME,
        num_train_epoch=NUM_TRAIN_EPOCH,
        base_model=model,
        batch_size=8
    )