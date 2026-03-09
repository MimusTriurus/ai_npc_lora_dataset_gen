from prefect import task, flow
from typing import List
import json
import uuid

@task(retries=3, retry_delay_seconds=10)
def generate_system_prompt() -> str:
    return ""

@task(retries=3, retry_delay_seconds=10)
def generate_usr_requests(n: int) -> List[str]:
    return []

@task(retries=3, retry_delay_seconds=10)
def generate_npc_answers():
    return []

@flow(name="lora-dataset-and-training")
def lora_pipeline(
    samples: int = 1000,
    lora_config: dict = None
):
    system_prompt = generate_system_prompt(samples)
    usr_requests = generate_usr_requests(1000)

    answers = generate_npc_answers()

    dataset_path = '1'
    return dataset_path

if __name__ == "__main__":
    lora_pipeline.deploy(
        name="lora-train-deployment",
        work_pool_name="local-pool"
    )


