#### Что делает скрипт (алгоритм)
#####Инициализация и подготовка:
- Загружает роли игроков (roles), шаблон system prompt, данные NPC из npc_description
- Находит нужного NPC по имени (npc_name) в actions_f_path
- Предварительно считает количество вхождений каждого action_name в ActionData (actions_count) — это нужно для формулы расчёта размера датасета

---

#### Для каждого ActionData шаблона:
- Парсинг сигнатуры — из ActionTemplate извлекается action_name и arg_names
- Раскрытие шаблона (build_action_template_params + render_template) — из шаблонного JSON генерируются все комбинации параметров (декартово произведение iterable-контейнеров []). Каждая комбинация — отдельный request_combination
- Расчёт roles_amount и requests_amount через calculate_roles_and_request_amount:
  - Учитывает current_actions_count (сколько раз этот action_name встречается в ActionData — т.е. сколько шаблонов дадут записи с этим именем)
  - Учитывает params_combination_count (количество раскрытых комбинаций)
  - Целевой размер датасета DATASET_SIZE_PER_ACTION = 4000
  - Подбирает roles и queries так, чтобы roles * queries * combinations * actions ≈ 4000

---

#### Двойной цикл role * request_combination:
- Формирует system_prompt (через Jinja) с указанием: шаблон запроса, роль игрока, описание NPC, ожидаемый action и параметры, количество запросов к генерации
- Вызывает LLM (OllamaHelper.generate) — просит сгенерировать num_requests перефраз в виде JSON-массива строк
- Валидирует результат: проверяет что каждый запрос содержит все целевые параметры (target_parameter.lower() in request.lower())
- До 5 попыток при ошибке парсинга/генерации

---

#### Сборка записи датасета для каждого сгенерированного запроса:
```
python{
    'usr_request': {
        "request": request,        # сгенерированная LLM перефраза
        "usr_state": usr_state,    # из NpcStateTemplate (уже раскрытый)
        "npc_state": npc_state,    # из UsrStateTemplate (уже раскрытый)
    },
    'npc_response': {
        "emotion": "",             # пусто — заполняется на шаге 3
        "answer": "",              # пусто — заполняется на шаге 3
        "think": "",
        "action": {
            "name": action_name,
            "parameters": args_dict  # из раскрытого шаблона
        }
    },
    'player_role': role
}
```
---

#### Сохранение в
*output_data/{npc_name}/0_generate_usr_requests/{action_name}.jsonl* **(append)**