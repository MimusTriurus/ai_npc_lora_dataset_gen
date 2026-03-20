from prefect import flow
import step_0_train.main as train_lora_adapter
import step_1_convert_to_gguf.main as convert_lora_to_gguf
import step_2_validation.main as validation_lora_adapter

@flow(name="train-lora-adapter-4-ue-npc")
def npc_lora_training_flow(
    unreal_commit: str,
    npc_name: str,
    flow_run_id: str = 'latest',
    base_model: str = "Qwen3-4B-Instruct-2507",
    num_train_epoch: int = 1,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    batch_size: int = 2,
):
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

if __name__ == '__main__':
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_NAME = 'trader'
    FLOW_RUN_ID = 'v1'

    npc_lora_training_flow(
        unreal_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id=FLOW_RUN_ID
    )