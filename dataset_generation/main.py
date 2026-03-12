from prefect import flow
from dotenv import load_dotenv
import dataset_generation.step_0_get_npc_desc.main as step_0_get_npc_desc
import dataset_generation.step_1_generate_usr_requests.main as step_1_generate_usr_requests
import dataset_generation.step_2_generate_sys_prompt.main as step_2_generate_sys_prompt
import dataset_generation.step_3_generate_npc_answers.main as step_3_generate_npc_answers
import dataset_generation.step_4_make_dataset.main as step_4_make_dataset

#@flow(name="lora-dataset-and-training")
def npc_lora_dataset_gen(
    git_commit: str,
    npc_name: str,
    flow_run_id: str,
):
    git_commit = git_commit[:7]

    step_0_get_npc_desc.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
    )

    step_1_generate_usr_requests.process(
        git_commit=git_commit,
        npc_name=npc_name,
        flow_run_id=flow_run_id,
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
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_NAME = 'trader'

    npc_lora_dataset_gen(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id='v1',
    )


