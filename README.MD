### Шаблонизация

#### Пример шаблона:
```
    {
        "ActionTemplate": "ShowItems({{ category[] }})",
        "Parameters": {
            "category": [
                "goods",
                "weapons",
                "ammo",
                "weapon upgrades",
                "medications"
            ],
            "goods": [
                "weapons, ammo, upgrades for weapon, medications"
            ],
            "weapons": [
                "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
            ],
            "ammo": [
                "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
            ],
            "weapon upgrades": [
                "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
            ],
            "medications": [
                "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
            ]
        },
        "RequestTemplate": "Do you have {{ category_it }} for purchase?",
        "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
        "NpcStateTemplate": "Store assortment: {{ join(Parameters[category_it]) }}"
    }
```

#### Операторы шаблонизации (расширенная Jinja)
- `{{ goods[] }}` - шаблон цикла for. Т.о. будет создано строк в кол-ве params. Можно имитировать вложенные циклы если такого типа параметров несколько в шаблоне
- `{{ goods_it }}` - конкретный элемент (суффикс _it добавляется к контейнеру параметров) из params на текущей итерации цикла
- `{{ rand_range(100, 500) }}` - вставляет число из диапазона
- `{{ join(goods) }}` - объединяет параметры из params в строку (разделитель - запятая)
- `{{ Parameters[goods_it] }}` получение списка параметров по ключу

#### Сгенерированные данные:

#### 1 - goods
```
{
    "ActionTemplate": "ShowItems(goods)",
    "Parameters": {
        "category": [
            "goods",
            "weapons",
            "ammo",
            "weapon upgrades",
            "medications"
        ],
        "goods": [
            "weapons, ammo, upgrades for weapon, medications"
        ],
        "weapons": [
            "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
        ],
        "ammo": [
            "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
        ],
        "weapon upgrades": [
            "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
        ],
        "medications": [
            "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
        ]
    },
    "RequestTemplate": "Do you have goods for purchase?",
    "UsrStateTemplate": "User has 132 gold",
    "NpcStateTemplate": "Store assortment: weapons, ammo, upgrades for weapon, medications"
}
```
#### 2 - weapons
```
{
    "ActionTemplate": "ShowItems(weapons)",
    "Parameters": {
        "category": [
            "goods",
            "weapons",
            "ammo",
            "weapon upgrades",
            "medications"
        ],
        "goods": [
            "weapons, ammo, upgrades for weapon, medications"
        ],
        "weapons": [
            "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
        ],
        "ammo": [
            "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
        ],
        "weapon upgrades": [
            "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
        ],
        "medications": [
            "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
        ]
    },
    "RequestTemplate": "Do you have weapons for purchase?",
    "UsrStateTemplate": "User has 251 gold",
    "NpcStateTemplate": "Store assortment: pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
}
```
#### 3 - ammo
```
{
    "ActionTemplate": "ShowItems(ammo)",
    "Parameters": {
        "category": [
            "goods",
            "weapons",
            "ammo",
            "weapon upgrades",
            "medications"
        ],
        "goods": [
            "weapons, ammo, upgrades for weapon, medications"
        ],
        "weapons": [
            "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
        ],
        "ammo": [
            "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
        ],
        "weapon upgrades": [
            "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
        ],
        "medications": [
            "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
        ]
    },
    "RequestTemplate": "Do you have ammo for purchase?",
    "UsrStateTemplate": "User has 332 gold",
    "NpcStateTemplate": "Store assortment: pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
}
```
#### 4 - weapon upgrades
```
{
    "ActionTemplate": "ShowItems(weapon upgrades)",
    "Parameters": {
        "category": [
            "goods",
            "weapons",
            "ammo",
            "weapon upgrades",
            "medications"
        ],
        "goods": [
            "weapons, ammo, upgrades for weapon, medications"
        ],
        "weapons": [
            "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
        ],
        "ammo": [
            "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
        ],
        "weapon upgrades": [
            "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
        ],
        "medications": [
            "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
        ]
    },
    "RequestTemplate": "Do you have weapon upgrades for purchase?",
    "UsrStateTemplate": "User has 257 gold",
    "NpcStateTemplate": "Store assortment: scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
}
```
#### 5 - medications
```
{
    "ActionTemplate": "ShowItems(medications)",
    "Parameters": {
        "category": [
            "goods",
            "weapons",
            "ammo",
            "weapon upgrades",
            "medications"
        ],
        "goods": [
            "weapons, ammo, upgrades for weapon, medications"
        ],
        "weapons": [
            "pistol, shotgun, revolver, sniper rifle, assault rifle, RPG"
        ],
        "ammo": [
            "pistol's ammo, shotgun's ammo, revolver's ammo, sniper rifle's ammo, assault rifle's ammo, RPG's ammo"
        ],
        "weapon upgrades": [
            "scope, extended_magazine, silencer, laser_sight, grip, recoil_reducer"
        ],
        "medications": [
            "bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
        ]
    },
    "RequestTemplate": "Do you have medications for purchase?",
    "UsrStateTemplate": "User has 355 gold",
    "NpcStateTemplate": "Store assortment: bandage, medkit, antidote, adrenaline_shot, painkillers, revive_kit"
}
```