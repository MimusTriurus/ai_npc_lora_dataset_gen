import re
import json
from collections import defaultdict

def build_system_prompt(data):
    action_map = {}  # { action_name: { param_name: set(values) } }

    for block in data:
        for action in block["ActionData"]:
            raw_name = action["ActionName"]

            # Убираем шаблонные параметры: CraftItem({{ material }}, {{ tool }}) → CraftItem
            clean_name = re.sub(r"\(\s*\{\{.*?\}\}\s*\)", "", raw_name)

            if clean_name not in action_map:
                action_map[clean_name] = defaultdict(set)

            # Собираем параметры по типам
            for param_name, values in action["Parameters"].items():
                for v in values:
                    action_map[clean_name][param_name].add(v)

    # Формируем SystemPrompt
    output = []

    for action_name, params_dict in action_map.items():
        param_list = list(params_dict.keys())

        # Заголовок действия
        output.append(f"{action_name}")
        output.append(f"   parameters: {json.dumps([f'<{p}>' for p in param_list])} where")

        # Условия для каждого параметра
        for p in param_list:
            output.append(f"      <{p}> is one of AllowedParameters_{p}")

        output.append("")

        # AllowedParameters
        for p, values in params_dict.items():
            output.append(f"AllowedParameters_{p}")
            for v in sorted(values):
                output.append(f"- {v}")
            output.append("")

    return "\n".join(output)

json_data = [
  {
    "ActionData": [
      {
        "ActionName": "CraftItem({{ material }}, {{ tool }})",
        "Parameters": {
          "material": ["iron", "wood", "stone"],
          "tool": ["hammer", "saw", "chisel"]
        },
        "RequestTemplate": "Could you craft a weapon using {{ material }} and {{ tool }}?",
        "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
        "NpcStateTemplate": "Materials: {{ material }}, Tools: {{ tool }}"
      },
      {
        "ActionName": "CraftItem({{ material }}, {{ tool }})",
        "Parameters": {
          "material": ["steel", "obsidian"],
          "tool": ["forge", "anvil"]
        },
        "RequestTemplate": "I need a special item made from {{ material }} using a {{ tool }}.",
        "UsrStateTemplate": "User has {{ rand_range(200, 800) }} gold",
        "NpcStateTemplate": "Workshop: {{ tool }}, Materials: {{ material }}"
      }
    ]
  }
]

result = build_system_prompt(json_data)
print(result)

