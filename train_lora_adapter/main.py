from prefect import flow
from prefect.context import get_run_context
import asyncio
import step_0_train.main as train_lora_adapter
import step_1_convert_to_gguf.main as convert_lora_to_gguf
import step_2_validation.main as validation_lora_adapter
import step_3_save_artifacts.main as save_artifacts

@flow(name="train-lora-adapter-4-ue-npc", log_prints=True)
async def npc_lora_training_flow(
    unreal_commit: str,
    npc_name: str,
    flow_run_id: str,
    base_model: str = "Qwen3-4B-Instruct-2507",
    num_train_epoch: int = 1,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    batch_size: int = 2,
):
    ctx = get_run_context()
    client = ctx.client
    run = ctx.flow_run
    run_name = f"train-lora-npc-{npc_name}-{flow_run_id}"
    await client.update_flow_run(
        flow_run_id=run.id,
        name=run_name
    )

    git_commit = unreal_commit[:7]

    train_lora_adapter.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        base_model=base_model,
        num_train_epoch=num_train_epoch,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
    )

    convert_lora_to_gguf.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        base_model=base_model,
    )

    validation_lora_adapter.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id
    )

    save_artifacts.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id
    )

@flow(name="validation-lora-adapter-4-ue-npc", log_prints=True)
async def npc_lora_validation_flow(
    unreal_commit: str,
    npc_name: str,
    flow_run_id: str,
):
    ctx = get_run_context()
    client = ctx.client
    run = ctx.flow_run
    run_name = f"validation-lora-npc-{npc_name}-{flow_run_id}"
    await client.update_flow_run(
        flow_run_id=run.id,
        name=run_name
    )

    git_commit = unreal_commit[:7]

    validation_lora_adapter.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id
    )

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'

    asyncio.run(
        npc_lora_validation_flow(
            unreal_commit=COMMIT,
            npc_name=NPC_NAME,
            flow_run_id=FLOW_RUN_ID
        )
    )
