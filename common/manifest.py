import json

from common.constants import DATA_DIR_NAME


class Manifest:
    def __init__(self, f_path):
        self.f_path = f_path
        with open(f_path, "r", encoding="utf-8") as f:
            self.obj = json.loads(f.read())

    def flow_dir_path(self):
        return f"{DATA_DIR_NAME}/{self.unreal_commit()}/{self.npc_name()}/{self.flow_run_id()}"

    def unreal_commit(self):
        return self.obj["initial_args"]['unreal_commit']

    def npc_name(self):
        return self.obj["initial_args"]['npc_name']

    def flow_run_id(self):
        return self.obj["initial_args"]['flow_run_id']

    def gguf_lora_f_path(self):
        gguf = self.obj['gguf']
        return f'{gguf["lora_f_path"]}'

    def gguf_model_f_path(self):
        gguf = self.obj['gguf']
        return f'{gguf["model_f_path"]}'

    def dataset_d_path(self):
        return f'{self.obj["dataset"]["dir_path"]}'

    def emb_dataset_mode(self):
        return self.obj["dataset"]["params"]["sentence2_mode"]

    def set_validation_results(self, validation_results: dict):
        self.obj["validation_results"] = validation_results

    def update(self):
        with open(self.f_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.obj, indent=4))

    def emb_request_prefix(self):
        return self.obj["training"]['query_prefix']