from prefect import flow
import dataset_generation.step_0_get_npc_desc.main as step_0_get_npc_desc
import dataset_generation.step_1_generate_usr_requests.main as step_1_generate_usr_requests
import dataset_generation.step_2_generate_sys_prompt.main as step_2_generate_sys_prompt
import dataset_generation.step_3_generate_npc_answers.main as step_3_generate_npc_answers
import dataset_generation.step_4_make_dataset.main as step_4_make_dataset

@flow(name="lora-dataset-generation")
def npc_lora_dataset_gen_flow(
    unreal_commit: str,
    npc_name: str,
    flow_run_id: str,
    use_npc_desc_gen: bool = True,
):
    git_commit = unreal_commit[:7]

    if use_npc_desc_gen:
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
    use_npc_desc_gen = False

    npc_lora_dataset_gen_flow(
        unreal_commit=COMMIT,
        npc_name=NPC_NAME,
        flow_run_id='v1',
        use_npc_desc_gen=use_npc_desc_gen
    )


