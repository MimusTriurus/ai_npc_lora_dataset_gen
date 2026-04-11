import warnings
import torch
import os
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import logging
from common.helpers import update_manifest
from prefect import task
from common.constants import *

warnings.filterwarnings("ignore")
logging.getLogger("trl").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["TRL_DISABLE_RICH"] = "1"
logging.getLogger("dis").setLevel(logging.CRITICAL)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

def analyze_token_lengths(dataset: Dataset, tokenizer: AutoTokenizer) -> int:
    lengths = []
    for example in dataset:
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        lengths.append(len(tokens))

    def round_up_to_even(x):
        return x if x % 2 == 0 else x + 1

    max_len = max(lengths)
    max_len = round_up_to_even(max_len)

    recommended_length = max(max_len, 1024)
    logger.info(f"Recommended max_seq_length: {recommended_length}")

    return recommended_length

@task(name="step_0_train_lora_adapter")
def process(
        git_commit: str,
        npc_name: str,
        flow_run_id: str,
        base_model: str = "Qwen3-4B-Instruct-2507",
        num_train_epoch: int = 1,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        batch_size: int = 2,
):
    model_path = f'models/{base_model}'

    if not os.path.exists(model_path):
        logger.info(f"Model not found locally, downloading from HuggingFace...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=f"Qwen/{base_model}", local_dir=model_path)

    dataset_dir = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{DATASET_DIR_NAME}'

    learning_rate = float(os.getenv('STEP_0_LEARNING_RATE', 5e-5))
    gradient_accumulation = int(os.getenv('STEP_0_GRADIENT_ACCUMULATION', 4))

    eval_strategy = os.getenv('STEP_0_EVAL_STRATEGY', 'no')
    eval_steps = int(os.getenv('STEP_0_EVAL_STEPS', 100))

    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"Model: {base_model}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Epochs: {num_train_epoch}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation}")
    logger.info("=" * 50 + "\n")

    dataset = load_dataset(
        "json",
        data_files={
            "train": f'{dataset_dir}/training/*.jsonl',
            "validation": f'{dataset_dir}/validation/*.jsonl',
        },
    )

    training_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Pad token set as eos_token: {tokenizer.eos_token}")

    tokenizer.padding_side = "right"

    logger.info("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = training_dataset.map(format_example, remove_columns=training_dataset.column_names)
    val_dataset_formatted = validation_dataset.map(format_example, remove_columns=validation_dataset.column_names)

    max_seq_length = analyze_token_lengths(train_dataset, tokenizer)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    sft_config = SFTConfig(
        output_dir=f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{LORA_DIR_NAME}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=num_train_epoch,

        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,

        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,

        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,

        #report_to="tensorboard",
    )

    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_formatted,
        data_collator=collator,
        args=sft_config,
        callbacks=[early_stopping_callback],
    )

    logger.info("Starting training...")
    trainer.train()

    save_dir = f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{LORA_DIR_NAME}/final_adapter"
    logger.info(f"Saving model to: {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    logger.info("Releasing GPU memory...")
    del trainer
    del model
    del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info("\nTraining completed successfully!")

    manifest_f_name = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/manifest.json'

    manifest = {
        'lora_training': {
            'model_name': base_model,
            'num_train_epoch': num_train_epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation': gradient_accumulation,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
        }
    }
    update_manifest(manifest_f_name, manifest)


if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id=FLOW_RUN_ID)
