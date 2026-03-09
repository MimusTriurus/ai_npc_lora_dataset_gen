## Интерактивный NPC 
### Идея: 
Игра (Unreal Engine 5) в которой игрок взаимодействует с NPC (MetaHuman) используя голос (SpeechToText)\текст.
- Для каждого NPC прописана (в DataAsset) его "биография", перечень действий которые он может совершать (procedure calling) и сэмплы голоса которые будут использованы при воспроизведения
- Вместе с текстом запроса игрока NPC получает информацию о состоянии игрока (usr_state) и контекст о состоянии игрового мира (npc_state)
  - npc_state инициализируется данными полученными из RAG системы. Для запроса игрока строятся embedding's и проводится поиск данных\тригера в "БД" игры. Эти данные добавляются в npc_state 
- NPC использует LLM для ответа на запрос.
  - NPC при ответе строго учитывает npc_state, usr_state
  - Ответ содержит:
    - информацию об эмоции которую NPC испытывает в ответ на запрос игрока
    - текст ответа NPC
- действие\action которое NPC выполняет в ответ на запрос игрока. Action является триггером для вызова события в игре
- NPC отвечает голосом (TextToSpeech)
- Губы NPC синхронизируются с аудио ответа (RealTime LipSync)

---

### Проблема:
- NPC LLM агент не всегда следует требуемому output format:
```
{
    "emotion": "Happy",
    "answer": "Sure! I've got some great weapon upgrades for you.",
    "action": {
        "name": "ShowItemsByCategory",
        "parameters": {
            "category": "weapon upgrades"
        }
    }
}
```
- NPC LLM агент не всегда вызывает корректный action в ответ на запрос игрока
- NPC LLM агент не всегда использует корректный набор параметров для action
- NPC LLM агент не всегда следует той роли которая для него прописана в игре или пытается отвечать на запросы которые находятся за рамками его области знаний
- NPC LLM агент не всегда учитывает npc_state, usr_state

### Решение
- Генерация Обучение LoRA адаптера 
---

### Репозитории:

- Проект **interactive_npc** на Unreal Engine 5 (git)
```
unreal-game/
│
├── Content/
│   └── DataAssets/
│       └── NPCs/
│           └── Trader/
│               └── npc_desc.uasset
│
├── Tools/
│   └── export_npc.py
│
└── .git/
```

- Проект **interactive_npc_ml** генерации датасета и обучения LoRA адаптера (git + dvc)
```
npc-ml/
│
├── flows/
│   ├── generate_dataset_flow.py
│   │   ├── 0_get_npc_desc_task.py
│   │   ├── 1_gen_usr_requests_task.py
│   │   ├── 2_gen_sys_prompt_task.py
│   │   ├── 3_gen_npc_answers_task.py
│   │   ├── 4_make_dataset.py
│   │   └── 5_export_dataset_to_dvc.py
│   └── train_lora_flow.py
│       ├── 0_prepare_env.py                # cloning necessary models, set-up and so on
│       ├── 1_train_lora.py
│       ├── 2_export_lora_to_gguf.py
│       ├── 3_evaluation.py
│       ├── 4_make_report.py
│       └── 5_export_lora_to_dvc.py
│
├── input_data/                     # gitignore
│       └── <npc_name>/
│           └── npc_description.json
│
├── output_data/                   
│   └── datasets/                   # under DVC
│       └── <npc_name>/
│           └── v_<hash>/
│               ├── temp_data/      # gitignore
│               │   ├── 1_gen_usr_requests/
│               │   │   └── <npc_action>.jsonl
│               │   ├── 2_gen_sys_prompt/
│               │   │   ├── actions_desc.json
│               │   │   └── system_prompt.txt
│               │   └── 3_gen_npc_answers/
│               │       └── <npc_action>.jsonl
│               ├── data/
│               └── dataset_metadata.json
│
├── models/                         # under DVC
│   ├── lora/
│   │   └── <npc_name>/
│   │       └── v_<hash>/
│   │           ├── weights/
│   │           └── lora_metadata.json
│   │
│   └── gguf/
│       └── <npc_name>/
│           └── v_<hash>/
│               ├── model.gguf
│               └── gguf_metadata.json
```

dataset_metadata_metadata.json
```
{
  "dataset_hash": "...",
  "unreal_commit": "...",
  "npc_ml_commit": "..."
}
```

lora_metadata.json
```
{
  "lora_hash": "...",
  "dataset_hash": "...",
  "unreal_commit": "...",
  "npc_ml_commit": "..."
  "base_model": "..."
}
```

lora_gguf_metadata.json
```
{
  "gguf_hash": "...",
  "lora_hash": "...",
  "dataset_hash": "...",
  "unreal_commit": "...",
  "npc_ml_commit": "..."
  "base_model": "..."
}
```

### Workflow:
#### 1. Создание описания NPC (точка входа workflow)
- В Unreal проекте **interactive_npc** есть набор объектов **DA_NPC_<имя npc>.uasset** типа *UDataAsset* с описанием NPC (бэкграунд, стиль речи, действия\actions которые он может совершать (procedure call))
- [Опционально] При коммите (пуше) изменений **DA_NPC_<имя npc>.uasset** (в определенную ветку?), срабатывает git-hook который:
  - определяет какие **DA_NPC_<имя npc>.uasset** были изменены 
  - используя Prefect API, добавляет в очередь (не запускает автоматически) flow генерации dataset'a (input: <имя npc> <хэш .git коммита>)
  
#### 2. Генерация датасета (Prefect flow):
- **task #0** - 0_get_npc_desc_task.py: 
  - pull'ит (или клонирует) изменения в unreal проекте **interactive_npc**, 
  - checkout на коммит с изменениями описания NPC, 
  - запускает unreal-commandlet который экспортирует описание NPC в json (**npc_description.json**) в папку
- **task #1** - 1_gen_usr_requests_task.py:   генерирует запросы игрока по шаблону для каждого NPC action
```
{
    "usr_request": {
        "request": "Do you have any goods for purchase? I'm just trying to stay alive, so anything useful would help.",
        "usr_state": "User has 328 gold",
        "npc_state": "Goods: weapons, ammo, upgrades for weapon, medications"
    },
    "npc_valid_action": {
        "emotion": "",
        "answer": "",
        "action": {
            "name": "ShowItemsByCategory",
            "parameters": {
                "category": "goods"
            }
        }
    }
}
```
- **task #2** - 2_gen_sys_prompt_task.py:    генерирует system_prompt для инференса модели по **npc_description.json**
- **task #3** - 3_gen_npc_answers_task.py:   генерирует ответы NPC на запросы игрока. заполняем полученными данными поля **emotion** и **answer** блока **npc_valid_action**
- **task #4** - 4_make_dataset.py:           формирует training и validation dataset для обучения LoRA адаптера
- **task $5** - 5_export_dataset_to_dvc.py:  версионирует и сохраняет dataset используя DVC (добавляет к датасету manifest.json для обратной трасировки)
```
{
  "dataset_hash": "...",
  "unreal_commit": "...",
}
```

#### 3. Обучение LoRA адаптера
- **task #0** - 0_prepare_env.py:            запускаем Prefect flow для обучения LoRA адаптера для NPC - указываем версию датасета
- **task #1** - 1_train_lora.py:             после обучения экспортируем LoRA адаптер в .gguf формат
- **task #2** - 2_export_lora_to_gguf.py:    преобразуем модель в .gguf формат для дальнейшего использования с llama.cpp
- **task #3** - 3_evaluation.py:             проверяем (относительно llm без LoRA адаптера например) ключевые метрики (json format, valid action, valid action args) на валидационном датасете 
- **task #4** - 4_make_report.py:            формируем отчет с результатами
- **task #5** - 5_export_lora_to_dvc.py:     версионируем LoRA адаптер при помощи DVC и push на Google drive


### Versioning:
