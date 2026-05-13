import warnings
import os
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from prefect import task
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CoSENTLoss
from transformers import EarlyStoppingCallback

from common.constants import (
    DATA_DIR_NAME,
    EMBEDDING_DATASET_DIR_NAME,
    LORA_EMBEDDING_DIR_NAME,
    MODELS_DIR_NAME,
)
from common.helpers import update_manifest

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


def _load_pairs_dataset(folder_glob: str):
    """Load *.jsonl pairs and keep only the columns CoSENTLoss expects."""
    ds = load_dataset("json", data_files=folder_glob, split="train")

    keep = {"sentence1", "sentence2", "score"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    # CoSENTLoss expects float labels in `score` column
    ds = ds.rename_column("score", "label")
    ds = ds.cast_column("label", ds.features["label"])
    return ds


@task(name="step_0_train_embedding_lora_adapter")
def process(
        git_commit: str,
        npc_name: str,
        flow_run_id: str,
        base_model: str = "BAAI/bge-base-en-v1.5",
        num_train_epoch: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        batch_size: int = 64,
):
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

    dataset_dir = (
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/'
        f'{EMBEDDING_DATASET_DIR_NAME}'
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
    logger.info("=" * 50 + "\n")

    train_ds = _load_pairs_dataset(f'{dataset_dir}/embedding_training/*.jsonl')
    val_ds = _load_pairs_dataset(f'{dataset_dir}/embedding_validation/*.jsonl')

    logger.info(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer(model_path)

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    # sentence-transformers >=3.0 routes add_adapter to the underlying transformer
    model.add_adapter(lora_config)

    # Print trainable params manually (ST doesn't expose print_trainable_parameters)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )

    loss = CoSENTLoss(model)

    output_dir = (
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/'
        f'{LORA_EMBEDDING_DIR_NAME}'
    )

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
        f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/manifest.json'
    )
    manifest = {
        'embedding_lora_training': {
            'model_name': base_model,
            'num_train_epoch': num_train_epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation': gradient_accumulation,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'seed': seed,
            'loss': 'CoSENTLoss',
            'pairs_train': len(train_ds),
            'pairs_val': len(val_ds),
        }
    }
    update_manifest(manifest_f_name, manifest)


if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")

    process(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id=FLOW_RUN_ID,
    )
