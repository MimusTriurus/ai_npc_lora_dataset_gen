import os

from prefect import flow
from dotenv import load_dotenv

@flow(name="lora-dataset-and-training")
def npc_lora_dataset_gen(
    git_commit: str,
    npc_name: str,
):
    load_dotenv('step_0_get_npc_desc/.env', override=True)
    import step_0_get_npc_desc.main
    step_0_get_npc_desc.main.process(
        git_commit=git_commit,
        npc_name=npc_name,
    )
    load_dotenv('step_1_generate_usr_requests/.env', override=True)
    import step_1_generate_usr_requests.main
    step_1_generate_usr_requests.main.process(
        git_commit=git_commit,
        npc_name=npc_name,
    )
    load_dotenv('step_2_generate_sys_prompt/.env', override=True)
    import step_2_generate_sys_prompt.main
    step_2_generate_sys_prompt.main.process(
        git_commit=git_commit,
        npc_name=npc_name,
    )
    load_dotenv('step_3_generate_npc_answers/.env', override=True)
    import step_3_generate_npc_answers.main
    step_3_generate_npc_answers.main.process(
        git_commit=git_commit,
        npc_name=npc_name,
    )
    load_dotenv('step_4_make_dataset/.env', override=True)
    import step_4_make_dataset.main
    step_4_make_dataset.main.process(
        git_commit=git_commit,
        npc_name=npc_name,
    )


if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"
    NPC_NAME = 'trader'

    npc_lora_dataset_gen(
        git_commit=COMMIT,
        npc_name=NPC_NAME,
    )


