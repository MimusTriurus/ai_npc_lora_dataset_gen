import json
import os
from collections import defaultdict
from typing import Dict, Tuple, List
from jinja2 import Environment
from prefect import task
from ullama_python.ullama import emotions

from common.constants import DATA_DIR_NAME, GEN_SYS_PROMPT_DIR_NAME, ACTION_FOR_IRRELEVANT_REQUESTS
from common.helpers import is_env_var_true, save_text_file, extract_nsloctext_value, parse_action_signature, \
    parse_actions_from_file

from common.ollama_helper import OllamaHelper

env = Environment()

GEN_ACTION_DESC_SP_F_PATH = os.getenv('STEP_2_GEN_ACTION_DESC_SP_F_PATH', '')
need_2_gen_action_desc = is_env_var_true('STEP_2_GENERATE_ACTION_DESC')
MODEL = os.getenv('STEP_2_OLLAMA_MODEL')

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

    irr_act_name, irr_act_desc = make_irrelevant_action_description()
    actions_params[irr_act_name] = {}
    action_desc[irr_act_name] = irr_act_desc

    for action_name, param_keys in actions_params.items():
        lines.append(f"{action_name}")
        if param_keys:
            d = {}
            for param in param_keys:
                d[param] = f"<{param}>"
            lines.append(f"   parameters: {json.dumps(d)} where")
        else:
            lines.append("   parameters: {}")

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
        cfg = {}
        helper = OllamaHelper(cfg)
        action_description, think = helper.generate(MODEL, sp)
        action['Description'] = action_description

def make_irrelevant_action_description() -> Tuple[str, str]:
    description = f"The user asks the NPC about topics completely unrelated to the NPC's role, abilities, or context - such as distant events, unrelated professions, abstract concepts, or impossible tasks - resulting in a request the NPC cannot meaningfully answer."
    return ACTION_FOR_IRRELEVANT_REQUESTS, description

def build_grammar(emotions: List[str], actions: List[dict], use_thinking_mode: bool = False) -> str:
    header = r'root ::= ThinkOrNothing nl nl Response' if use_thinking_mode else r'root ::= Response'

    thinking_rules = r'''
ThinkOrNothing ::= ThinkBlock | ""
ThinkBlock ::= "<think>" ThinkText "</think>"
Sentence ::= ([^.<] | "<" [^/])* "."
ThinkText ::= Sentence | Sentence Sentence | Sentence Sentence Sentence
''' if use_thinking_mode else r''

    common_rules = r'''
nl ::= "\n"
Action ::= "{" ws "\"name\":" ws actions "," ws "\"parameters\":" ws dict "}"
Response ::= "{" ws "\"emotion\":" ws emotions "," ws "\"answer\":" ws string "," ws "\"action\":" ws Action "}"
string ::= "\"" ([^"]*) "\""
ws ::= [ \t\n]*
kv ::= string ws ":" ws string
dict ::= "{" ws "}" | "{" ws kv ("," ws kv)* ws "}"
    '''

    emotions_rule = "emotions ::= " + " | ".join([rf'"\"{e}\""' for e in emotions])
    actions_rule = "actions ::= " + " | ".join([rf'"\"{a["name"]}\""' for a in actions])

    result = f"# GBNF Grammar\n{header}{thinking_rules}{common_rules}\n{emotions_rule}\n{actions_rule}"
    return result

@task(name="step_2_generate_sys_prompt_and_actions_description")
def process(git_commit: str, npc_name: str, flow_run_id: str):
    NPC_DESC_F_PATH = f'{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/description.json'
    INFERENCE_SP_F_PATH = os.getenv("STEP_2_INFERENCE_SP_F_PATH")

    SYSTEM_PROMPT_TEMPLATE = ''
    with open(INFERENCE_SP_F_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_TEMPLATE += f.read()
    sp_template = env.from_string(SYSTEM_PROMPT_TEMPLATE)

    with open(NPC_DESC_F_PATH, "r", encoding="utf-8") as f:
        npcs_desc = json.load(f)
        npcs_desc['Description'] = extract_nsloctext_value(npcs_desc['Description'])
        if is_env_var_true('GENERATE_ACTION_DESC'):
            generate_action_description(npcs_desc)
        actions_rules = build_actions_rules(npcs_desc)
        params = {
            "npc": npcs_desc['Description'],
            "actions_rules": actions_rules
        }
        sp = sp_template.render(params)

        actions_desc: Dict[str, str] = dict()
        for action in npcs_desc["ActionData"]:
            action_name, action_args = parse_action_signature(action["ActionTemplate"])
            action_desc = action["Description"]
            actions_desc[action_name] = action_desc

        irrelevant_action_name, irrelevant_action_desc = make_irrelevant_action_description()
        actions_desc[irrelevant_action_name] = irrelevant_action_desc
        actions_desc_f_path = f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_SYS_PROMPT_DIR_NAME}"

        # ACTIONS DESCRIPTION
        save_text_file(
            folder_path=actions_desc_f_path,
            filename="actions_desc.json",
            content=json.dumps(actions_desc, indent=2)
        )
        # SYSTEM_PROMPT
        save_text_file(
            folder_path=f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_SYS_PROMPT_DIR_NAME}",
            filename="system_prompt.txt",
            content=sp
        )

        # GRAMMAR
        actions = parse_actions_from_file(f'{actions_desc_f_path}/actions_desc.json')
        grammar_str = build_grammar(
            emotions=emotions,
            actions=actions,
            use_thinking_mode=False
        )
        save_text_file(
            folder_path=f"{DATA_DIR_NAME}/{git_commit}/{npc_name}/{flow_run_id}/{GEN_SYS_PROMPT_DIR_NAME}",
            filename="grammar.txt",
            content=grammar_str
        )

if __name__ == "__main__":
    COMMIT = "60e7a243ce941bd02e08429d4dbbdaecea1ca076"[:7]
    NPC_NAME = "trader"
    exit(process(git_commit=COMMIT, npc_name=NPC_NAME, flow_run_id='v1'))