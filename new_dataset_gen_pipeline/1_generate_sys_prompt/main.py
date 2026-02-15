import json
import os
from collections import defaultdict
from jinja2 import Environment
from common.helpers import is_env_var_true, save_text_file, extract_nsloctext_value, parse_action_signature
from common.ollama_helper import OllamaHelper, OLLAMA_HOST, MODEL

env = Environment()

GEN_ACTION_DESC_SP_F_PATH = os.getenv('GEN_ACTION_DESC_SP_F_PATH', '')
need_2_gen_action_desc = is_env_var_true('GENERATE_ACTION_DESC')

def build_actions_rules(action_data):
    actions_params = {}
    action_desc = {}
    param_groups = defaultdict(set)

    for action in action_data['ActionData']:
        raw_name = action["ActionTemplate"]
        action_name, arg_names = parse_action_signature(raw_name)
        actions_params[action_name] = arg_names
        action_desc[action_name] = action["Description"] if need_2_gen_action_desc else ''
        for key, values in action["Parameters"].items():
            if key in arg_names:
                param_groups[key].update(values)

    lines = []

    for action_name, param_keys in actions_params.items():
        lines.append(f"{action_name}")
        if param_keys:
            lines.append(f"   parameters: {json.dumps([f'<{p}>' for p in param_keys])} where")
        else:
            lines.append(f"   parameters: []")

        for p in param_keys:
            lines.append(f"      <{p}> is one of AllowedParameters_{p}")
        if action_desc[action_name]:
            lines.append(f"   description: {action_desc[action_name]}")
        lines.append("")

    for key, values in param_groups.items():
        lines.append(f"AllowedParameters_{key}:")
        for v in sorted(values):
            lines.append(f"- {v}")
        lines.append("")

    return "\n".join(lines)

def merge_actions(action_data) -> list:
    merged = {}

    for action in action_data["ActionData"]:
        template = action["ActionTemplate"]

        if template not in merged:
            merged[template] = {
                "ActionTemplate": template,
                "Parameters": {k: list(v) for k, v in action["Parameters"].items()},
                "RequestTemplate": [action["RequestTemplate"]],
                "UsrStateTemplate": action["UsrStateTemplate"],
                "NpcStateTemplate": action["NpcStateTemplate"]
            }
        else:
            for key, values in action["Parameters"].items():
                if key in merged[template]["Parameters"]:
                    merged[template]["Parameters"][key].extend(values)
                else:
                    merged[template]["Parameters"][key] = list(values)

            merged[template]["RequestTemplate"].append(action["RequestTemplate"])

    result = list(merged.values())
    return result


def generate_action_description(npc_data: dict):
    if not npc_data:
        return

    SYSTEM_PROMPT_TEMPLATE = ''
    with open(GEN_ACTION_DESC_SP_F_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_TEMPLATE += f.read()

    npc_data["ActionData"] = merge_actions(npc_data)

    for action in npc_data["ActionData"]:
        params = {
            "npc": npc_data['Description'],
            "action": json.dumps(action, indent=2)
        }
        sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)
        sp = sp_template.render(params)

        helper = OllamaHelper(OLLAMA_HOST)
        action_description, think = helper.generate(MODEL, sp)
        action['Description'] = action_description

if __name__ == "__main__":
    NPC_DESC_F_PATH = os.getenv("NPC_DESC_F_PATH")
    INFERENCE_SP_F_PATH = os.getenv("INFERENCE_SP_F_PATH")

    SYSTEM_PROMPT_TEMPLATE = ''
    with open(INFERENCE_SP_F_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_TEMPLATE += f.read()
    sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)

    with open(NPC_DESC_F_PATH, "r", encoding="utf-8") as f:
        npcs_desc = json.load(f)
        for npc_data in npcs_desc:
            desc = npc_data["Description"]
            npc_data['Description'] = extract_nsloctext_value(npc_data['Description'])
            if is_env_var_true('GENERATE_ACTION_DESC'):
                generate_action_description(npc_data)
            actions_rules = build_actions_rules(npc_data)
            params = {
                "npc": npc_data['Description'],
                "actions_rules": actions_rules
            }
            sp = sp_template.render(params)

            save_text_file(
                folder_path=f"output_data/{npc_data['Name']}/1_generate_usr_requests",
                filename="system_prompt.txt",
                content=sp
            )