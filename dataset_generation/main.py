import os

from prefect import flow
from prefect.context import get_run_context
import asyncio
import dataset_generation.step_0_get_npc_desc.main as step_0_get_npc_desc
import dataset_generation.step_1_generate_usr_requests.main as step_1_generate_relevant_usr_requests
import dataset_generation.step_1_generate_usr_requests.gen_irrelevant_requests as step_1_generate_irrelevant_usr_requests
import dataset_generation.step_2_generate_sys_prompt.main as step_2_generate_sys_prompt
import dataset_generation.step_3_generate_npc_answers.main as step_3_generate_npc_answers
import dataset_generation.step_4_make_dataset.main as step_4_make_dataset

@flow(name="lora-dataset-generation", log_prints=True)
async def npc_lora_dataset_gen_flow(
    unreal_commit: str,
    npc_name: str,
    flow_run_id: str,
    dataset_size_per_action: int
):
    ctx = get_run_context()
    client = ctx.client
    run = ctx.flow_run
    run_name = f"gen-dataset-npc-{npc_name}-{flow_run_id}"
    await client.update_flow_run(
        flow_run_id=run.id,
        name=run_name
    )

    git_commit = unreal_commit[:7]

    step_0_get_npc_desc.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
    )

    step_1_generate_relevant_usr_requests.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        dataset_size_per_action=dataset_size_per_action,
    )

    step_1_generate_irrelevant_usr_requests.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
        dataset_size_per_action=dataset_size_per_action,
    )

    step_2_generate_sys_prompt.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
    )

    step_3_generate_npc_answers.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
    )

    step_4_make_dataset.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
    )


if __name__ == "__main__":
    COMMIT = os.getenv("COMMIT")
    NPC_NAME = os.getenv("NPC_NAME")
    FLOW_RUN_ID = os.getenv("FLOW_RUN_ID")
    DATASET_SIZE_PER_ACTION = os.getenv('DATASET_SIZE_PER_ACTION', 100)

    asyncio.run(
        npc_lora_dataset_gen_flow(
            unreal_commit=COMMIT,
            npc_name=NPC_NAME,
            flow_run_id=FLOW_RUN_ID,
            dataset_size_per_action=DATASET_SIZE_PER_ACTION,
        )
    )


