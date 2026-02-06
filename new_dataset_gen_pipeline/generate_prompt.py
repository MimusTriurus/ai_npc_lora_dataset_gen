import re

# очистка string поля от меток Unreal FText
def extract_nsloctext_value(text: str) -> str:
    match = re.search(r'NSLOCTEXT\([^,]+,\s*[^,]+,\s*\"(.*)\"\)', text)
    if not match:
        return text

    value = match.group(1)

    value = value.encode('utf-8').decode('unicode_escape')

    return value


def generate_system_prompt(json_data: dict) -> str:
    if not json_data:
        return ""

    npc = json_data

    all_parameters = set()
    for action in npc["ActionData"]:
        all_parameters.update(action["Parameters"])

    param_groups = {}
    for action in npc["ActionData"]:
        params = tuple(sorted(action["Parameters"]))

        if params not in param_groups:
            param_groups[params] = []
        param_groups[params].append(action)

    unique_param_sets = {}
    for params_tuple in param_groups.keys():
        if params_tuple:
            param_set_name = f"AllowedParameters_{len(unique_param_sets) + 1}"
            unique_param_sets[params_tuple] = param_set_name

    actions = []
    action_counter = 1

    for action in npc["ActionData"]:
        action_name = action["ActionName"]
        params = action["Parameters"]
        description = action["Description"]

        if not params:
            param_format = "parameters: []"
        else:
            params_tuple = tuple(sorted(params))
            param_set_name = unique_param_sets[params_tuple]
            param_format = f'parameters: ["<param>"] where <param> is one of {param_set_name}'

        action_counter += 1

        actions.append({
            "name": action_name,
            "params": param_format,
            "description": description
        })

    prompt_parts = ["Allowed actions and STRICT parameter rules:"]

    for i, action in enumerate(actions, 1):
        prompt_parts.append(f"{i}. {action['name']}")
        prompt_parts.append(f"   {action['params']}")
        prompt_parts.append(f"   description: {action['description']}")

    for params_tuple, param_set_name in unique_param_sets.items():
        prompt_parts.append(f"\n{param_set_name}:")
        for param in sorted(params_tuple):
            prompt_parts.append(f"- {param}")

    return "\n".join(prompt_parts)


if __name__ == "__main__":
    import json

    with open("npc_description.json", "r", encoding="utf-8") as f:
        npcs_desc = json.load(f)
        for npc_data in npcs_desc:
            desc = npc_data["Description"]
            npc_data['Description'] = extract_nsloctext_value(npc_data['Description'])
            prompt = generate_system_prompt(npc_data)
            print(prompt)