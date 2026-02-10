import json

data = {
    "ActionData": [
        {
            "ActionTemplate": "SellItem({{ item }})",
            "Parameters": {
                "item": ["pistol", "rifle", "shotgun", "revolver", "sniper_rifle", "rocket_launcher"]
            },
            "RequestTemplate": "Could you sell me the {{ item }} for defence from monsters.",
            "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
            "NpcStateTemplate": "Goods: {{ item }}, Cost: {{ rand_range(50, 99) }}, Amount: {{ rand_range(1, 10) }}"
        },
        {
            "ActionTemplate": "DontHaveEnoughMoney({{ item }})",
            "Parameters": {
                "item": ["pistol", "rifle", "shotgun", "revolver", "sniper_rifle", "rocket_launcher"]
            },
            "RequestTemplate": "Could you sell me the {{ item }} for defence from monsters.",
            "UsrStateTemplate": "User has {{ rand_range(0, 49) }} gold",
            "NpcStateTemplate": "Goods: {{ item }}, Cost: {{ rand_range(50, 99) }}, Amount: {{ rand_range(1, 10) }}"
        },
        {
            "ActionTemplate": "SoldOut({{ item }})",
            "Parameters": {
                "item": ["pistol", "rifle", "shotgun", "revolver", "sniper_rifle", "rocket_launcher"]
            },
            "RequestTemplate": "Could you sell me the {{ item }} for defence from monsters.",
            "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
            "NpcStateTemplate": "Goods: {{ item }}, Cost: {{ rand_range(50, 99) }}, Amount: 0"
        },
        {
            "ActionTemplate": "SellItem({{ item }})",
            "Parameters": {
                "item": ["pistol's ammo", "shotgun's ammo", "rifle's ammo", "revolver's ammo", "sniper_rifle's ammo",
                         "rocket_launcher's ammo"]
            },
            "RequestTemplate": "Could you sell me the {{ item }} for my weapon?",
            "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
            "NpcStateTemplate": "Goods: {{ item }}, Cost: {{ rand_range(10, 50) }}, Amount: {{ rand_range(1, 100) }}"
        }
    ]
}

merged = {}

for action in data["ActionData"]:
    template = action["ActionTemplate"]

    if template not in merged:
        # Создаем новую запись, превращая RequestTemplate в массив
        merged[template] = {
            "ActionTemplate": template,
            "Parameters": {k: list(v) for k, v in action["Parameters"].items()},
            "RequestTemplate": [action["RequestTemplate"]],
            "UsrStateTemplate": action["UsrStateTemplate"],
            "NpcStateTemplate": action["NpcStateTemplate"]
        }
    else:
        # 1. Объединяем списки в Parameters
        for key, values in action["Parameters"].items():
            if key in merged[template]["Parameters"]:
                merged[template]["Parameters"][key].extend(values)
            else:
                merged[template]["Parameters"][key] = list(values)

        # 2. Добавляем RequestTemplate в массив
        merged[template]["RequestTemplate"].append(action["RequestTemplate"])

        # UsrStateTemplate и NpcStateTemplate не трогаем (остаются от первого)

# Превращаем словарь обратно в список
result = {"ActionData": list(merged.values())}

# Вывод результата
print(json.dumps(result, indent=2, ensure_ascii=False))