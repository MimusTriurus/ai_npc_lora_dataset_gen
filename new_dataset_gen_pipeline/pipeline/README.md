Оцени концепцию. Критика. Вопросы? Предложи улучшения.

# Dataset generation and LoRA adapter training (ML Pipeline Automation или Data-driven Model Specialization)
### Задача: реализовать игрового NPC который взаимодействует с игроком:
- отвечает на запросы игрока исходя из контекста (usr_state, npc_state)
- procedure call с набором параметров в ответ на запрос игрока

### Проблема: instruct llm не всегда следуют инструкции: вызывают не те action, не с нужным набором параметров, нарушают структуры вывода (не json формат)

### Решение: сгенерировать датасет используя описание NPC и обучить LoRA адаптер

---

### Как это может выглядеть с точки зрения игрока?
- Точка входа - таблица **npc_description** (UDataTable). Содержит описание разных NPC (специализация, черты характера, список actions и параметров).
- В Unreal Engine добавляется запись в npc_description с описанием игрового NPC
- Изменения Unreal проекта пушатся в отдельную ветку git репозитория
- срабатывает git-hook если появилась новая запись (или изменились ключевые поля старой записи - дрейф данных(?)) в DataTable
- запускается выполнение графа задач в Prefect:
  - генерируется датасет
    - положительные примеры
      - system_prompt на базе **npc_description**
      - запросы игрока на основе примера запроса для Action из **npc_description**
      - ответы npc с учетом расчитаного intention игрока и ожидаемого action (action и параметры для него уже были заданы в шаблоне)
    - негативные примеры: 
      - out-of-scope запросы
      - запросы игрока не совпадающие с usr_state\npc_state (см. пункт "Пример описания NPC из **npc_description** (сериализованного в json)")
      - ...
  - обучается LoRA адаптер (см. prefect flow)
  - валидация результата обучения (action accuracy + parameter accuracy + format validity)(Coherence LLM judge опционально)
    - Опционально: canary-обучение нескольких LoRA и проверка нескольких вариантов LoRA-адаптера на одном и той же базовой модели и датасете, с последующим автоматическим отбором лучшего варианта
  - сохранение результата (.gguf LoRA адаптер)
---

###  Пример описания NPC из **npc_description** (сериализованного в json)
Пример описания NPC - по сути, формальная декларативная схема на базе расширенного Jinja template.
```
[
  {
    "Name": "npc_trader",
    "Description": "NAME: The Merchant SPECIALIZATION: Weapons, upgrades, medications, ammo TRAITS: Charismatic, loves making deals, always smiling.")",
    "ActionData": [
      {
        "ActionTemplate": "SellItem({{ item[] }})",
        "Parameters": {
          "item": [
            "scope",
            "extended_magazine",
            "silencer",
            "laser_sight",
            "grip",
            "recoil_reducer"
          ]
        },
        "RequestTemplate": "Could you sell me the {{ item_it }} for my weapon? I want to improve it.",
        "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
        "NpcStateTemplate": "Item: {{ item_it }}, Cost: {{ rand_range(10, 50) }} gold, Amount: {{ rand_range(1, 100) }}"
      },
      {
        "ActionTemplate": "NotEnoughGoldForBuy({{ item[] }})",
        "Parameters": {
          "item": [
            "scope",
            "extended_magazine",
            "silencer",
            "laser_sight",
            "grip",
            "recoil_reducer"
          ]
        },
        "RequestTemplate": "Could you sell me the {{ item_it }} for my weapon? I want to improve it.",
        "UsrStateTemplate": "User has {{ rand_range(0, 49) }} gold",
        "NpcStateTemplate": "Item: {{ item_it }}, Cost: {{ rand_range(50, 99) }} gold, Amount: {{ rand_range(1, 10) }}"
      },
      {
        "ActionTemplate": "SoldOut({{ item[] }})",
        "Parameters": {
          "item": [
            "scope",
            "extended_magazine",
            "silencer",
            "laser_sight",
            "grip",
            "recoil_reducer"
          ]
        },
        "RequestTemplate": "Could you sell me the {{ item_it }} for my weapon? I want to improve it.",
        "UsrStateTemplate": "User has {{ rand_range(100, 500) }} gold",
        "NpcStateTemplate": "Item: {{ item_it }}, Cost: {{ rand_range(50, 99) }} gold, Amount: 0"
      }
    ]
  }
]
```

Используется следующий синтаксис:
```
{{ category[] }}
{{ category_it }}
{{ Parameters[key] }}
{{ rand_range() }}
```

- **[ ]** - означает итерацию (цикл for)
- **_it** - означает текущий элемент
- **Parameters[...]** - означает доступ к значению по ключу
- rand_rage - подставляет числовое значение из диапазона

Формула расчета количества генераций по шаблону для каждого Action из ActionData
D = (n₁ * n₂ * ... * nₖ) * p * r
Где:
- n₁ ... nₖ — количество элементов в каждом iterable-контейнере **[]**
(каждый вложенный цикл даёт отдельный множитель)
- k - количество контейнеров [] в шаблоне
- p - количество личностей
- r - количество вариантов RequestTemplate на одну личность

Параметры **p** и **r** расчитываются исходя из желаемого размера датасета

##### Описание полей шаблона:
- **ActionTemplate** - сигнатура функции которую может вызывать NPC
- **Parameters** - пространство допустимых значений используемое для генерации данных для датасета по шаблону
- **RequestTemplate** - шаблон запроса игрока (для каждого запроса построенного по шаблону LLM генерует аналоги\парафразы\unclear-requests запросы от лица разных (60+) личностей каждая со своим бэкграундом).
- **UsrStateTemplate** - описывает состояние игрока на момент запроса (формируется исходя из запроса игрока при помощи **RAG**) 
- **NpcStateTemplate** - описывает состояние npc на момент запроса. По сути это "память" NPC. (формируется исходя из запроса игрока при помощи **RAG**)

---

### Генерация датасета
- Генерация запросов игрока по шаблону:
  - на основании **npc_description** формируется **system_prompt** (который будет использоваться при SFT обучении и инференсе модели) с описанием npc и описанием каждого Action (генерируется LLM)  
  - на основании **npc_description** формируется базовый набор записей датасета для каждого Action из ActionaData. npc_response уже содержит ожидаемый от NPC action и набор parameters. 
##### Примеры записей:
```
{
    "usr_request": {
        "request": "Sell me the pistol.",
        "usr_state": "User has 456 gold",
        "npc_state": "Title: Pistol, Cost: 100 gold, Amount: 1"
    },
    "npc_response": {
        "emotion": "",
        "answer": "",
        "action": {
            "name": "SellItem",
            "parameters": {
                "weapon": "pistol"
            }
        }
    }
}
```

  - для каждого **user_request** LLM переформулирует поле **request** в соответствии с заданой ролью игрока (всего доступно 60+ ролей). Поля **usr_state**, **npc_state** не меняются
##### Пример роли игрока:
```
"player_role": {
    "name": "Veteran Soldier",
    "description": "A battle-tested professional with years of combat experience.",
    "speech_style": "direct, confident, tactical"
}
```
  - для каждого **user_request** LLM генерирует **npc_response** заполняя поля **emotion** и **answer**
```
{
    "usr_request": {
        "request": "Sell me the pistol.",
        "usr_state": "User has 456 gold",
        "npc_state": "Title: Pistol, Cost: 100 gold, Amount: 1"
    },
    "npc_valid_action": {
        "emotion": "Happy",
        "answer": "Sure thing, Veteran! Take this pistol.",
        "action": {
            "name": "SellItem",
            "parameters": {
                "weapon": "pistol"
            }
        }
    }
}
```

---

### Prefect flow. Dataset generation + LoRA training + evaluation:
1. generate_user_requests
- input:
  - json file (описание npc и actions которые он может инициировать в ответ на запрос игрока)
- output:
  - набор json file для каждого action Action.jsonl (содержит список запросов игрока и ожидаемое от npc action с параметрами)
2. generate_system_prompt
- input:
  - json file (описание npc и actions которые он может инициировать в ответ на запрос игрока)
- output:
  - system_prompt.txt для inference модели + LoRA адаптера
  - actions_desc.json - описание для каждого action (намерение игрока)
3. generate_npc_answers
- input:
  - json file (описание npc и actions которые он может инициировать в ответ на запрос игрока)
  - набор json file для каждого action Action.jsonl (содержит список запросов игрока и ожидаемое от npc action с параметрами)
  - actions_desc.json - описание для каждого action (намерение игрока)
- output:
  - набор json file для каждого action Action.jsonl (к списку запросов игрока добавлены ответы NPC на эти запросы
4. generate_dataset
- input:
  - system_prompt.txt 
  - набор json file для каждого action Action.jsonl
- output:
  - набор json file для каждого action Action.jsonl в формате для обучения LoRA адаптера (training\validation)
5. train_lora_adapter
- input:
  - json с гиперпараметрами для обучения LoRA адаптера
  - название модели
  - путь к training датасету
  - путь к валидационному датасету
- output:
  - папка с файлами LoRA адаптера
6. convert_to_gguf_format
- input:
  - папка с файлами LoRA адаптера
- output:
  - .gguf файл
6. lora_adapter_evaluation
- input:
  - путь к валидационному датасету
- output:
  - файл с метриками

---

### Пример запроса игрока к LLM NPC:
```
{
    "context": "Item: laser_sight. Price: 299 gold. Amount: 8",
    "state_of_user": "User has 3072 gold.",
    "request_of_user": "How about you drop me a piece of that laser_sight for my gear? Just a quick run, nothing fancy."
}
```

### Пример ответа LLM NPC на запрос игрока:
```
{
    "emotion": "Happy",
    "answer": "Sure! That laser sight fits your run perfectly!",
    "action": {
        "name": "SellItem",
        "parameters": [
            "laser_sight"
        ]
    }
}
```

