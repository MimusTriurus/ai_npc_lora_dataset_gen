# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end pipeline for generating LoRA fine-tuning datasets and training character-specific LLM adapters for NPCs in Unreal Engine 5. Two independent Prefect-orchestrated workflows:

1. **Dataset Generation** (`dataset_generation/`) — synthesizes conversation training data for a named NPC
2. **LoRA Training** (`train_lora_adapter/`) — trains, converts, validates, and saves a LoRA adapter

## Running the Pipelines

Each workflow is triggered via its `main.py` with environment variables set in the corresponding `.env` file:

```bash
# From repo root
python dataset_generation/main.py
python train_lora_adapter/main.py
```

Key env vars that identify a run (used in artifact paths and versioning):
- `COMMIT` — git hash of the UE project being targeted
- `NPC_NAME` — name of the character (e.g. `trader`)
- `FLOW_RUN_ID` — logical run identifier (e.g. `v1`)

### Prefect Setup (one-time)

See `PREFECT.md` for full setup. Quick start:

```bash
prefect server start                          # Dashboard at http://127.0.0.1:4200
prefect work-pool create local --type process
prefect worker start --pool local
prefect deploy -n dataset-generation --no-prompt
prefect deploy -n lora-adapter-training --no-prompt
```

## Architecture

### Dataset Generation Pipeline (`dataset_generation/`)

Steps run sequentially via Prefect tasks in `main.py`:

| Step | Directory | Purpose |
|------|-----------|---------|
| 0 | `step_0_get_npc_desc/` | Clone UE project, extract NPC description JSON |
| 1 | `step_1_generate_usr_requests/` | Template-based generation of user queries and irrelevant requests |
| 2 | `step_2_generate_sys_prompt/` | Build system prompt from NPC description |
| 3 | `step_3_generate_npc_answers/` | Ollama LLM generates valid NPC responses (JSONL) |
| 4 | `step_4_make_dataset/` | Stratified train/validation split by action name |
| 5 | `step_5_make_knowledge_base/` | Build RAG knowledge base |
| 6 | `step_6_validate_knowledge_base/` | Validate knowledge base retrieval |

### LoRA Training Pipeline (`train_lora_adapter/`)

| Step | Directory | Purpose |
|------|-----------|---------|
| 0 | `step_0_train/` | SFT with HuggingFace `transformers` + PEFT |
| 1 | `step_1_convert_to_gguf/` | Convert adapter to GGUF via `llama.cpp` |
| 2 | `step_2_validation/` | Evaluate base vs. adapted model on validation set |
| 3 | `step_3_save_artifacts/` | Version and upload artifacts to MinIO (S3) |

### Shared Utilities (`common/`)

- `data_classes.py` — `Action`, `Question`, `PlayerRole` dataclasses (core schema)
- `helpers.py` — JSONL I/O, Jinja2 templating, dataset manifest management
- `inference.py` — Unified inference wrapper (Ollama or ULlama backends)
- `ollama_helper.py` / `ullama_helper.py` — Backend-specific clients
- `storage.py` — MinIO/S3 artifact upload/download
- `metrics_plot_generation.py` — Validation metric visualizations
- `training_results_report_generation.py` — Markdown report generation

### Validation Metrics

The validation step (`train_lora_adapter/step_2_validation/`) measures:
- JSON parse success rate
- Action correctness (did the NPC call the right action?)
- Parameter matching accuracy

Results are saved to `reports/report.md` and metric plots. The validation dataset is stratified by action name, with equal samples per action per parameter variant.

## External Dependencies

- **Ollama** — must be running locally for dataset generation (model configured via `STEP_1_OLLAMA_MODEL`)
- **llama.cpp** — compiled locally; path set via `STEP_1_LLAMA_CPP_DIR`
- **MinIO** — optional S3-compatible storage for artifact versioning
- **Unreal Engine 5** — required by Step 0 to export NPC data; path via `STEP_0_UE_DIR_PATH`
- **CUDA** — required for training step

## Key Configuration Files

- `dataset_generation/.env` — env vars for generation pipeline
- `train_lora_adapter/.env` — env vars for training pipeline
- `prefect.yaml` — Prefect deployment definitions for both workflows
- `common/constants.py` — canonical directory/file name constants used across all steps
