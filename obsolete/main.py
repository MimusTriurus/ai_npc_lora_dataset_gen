

import json
import random
import time

from common.helpers import actions_dict_to_signatures, make_actions_str, PromptBuilder
from common.ollama_helper import *

DEFAULT_NPC_DESCRIPTION = '''
<npc_description> 
NAME: The Merchant 
RACE: Human 
SPECIALIZATION: Weapons, upgrades, rare items 
BACKGROUND: A mysterious trader from Resident Evil 4 who appears in unexpected places to sell the player essential gear for survival. 
TRAITS: Charismatic, loves making deals, always smiling, legendary catchphrase "What are ya buying"? 
</npc_description>
'''

# Сколько примеров на сценарий
N_TRADE_WEAPON   = int(os.getenv('N_TRADE_WEAPON',   '800'))
N_AMMO           = int(os.getenv('N_AMMO',           '400'))
N_UPGRADES       = int(os.getenv('N_UPGRADES',       '300'))
N_SHOW_ALL       = int(os.getenv('N_SHOW_ALL',       '200'))
N_OFF_ROLE       = int(os.getenv('N_OFF_ROLE',       '300'))

# Пауза между запросами (мс), чтобы не перегружать Ollama
DELAY_MS = int(os.getenv('DELAY_MS', '100'))

NPC_DESCRIPTION = os.getenv('NPC_DESCRIPTION', DEFAULT_NPC_DESCRIPTION)

WEAPONS = [
    "pistol",
    "shotgun",
    "rifle",
    "SMG",
    "sniper_rifle",
    "rocket_launcher"
]

WEAPON_UPGRADES = [
    "scope",                # оптический прицел
    "extended_magazine",    # увеличенный магазин
    "silencer",             # глушитель
    "laser_sight",          # лазерный целеуказатель
    "grip",                 # эргономичная рукоять
    "recoil_reducer",       # компенсатор отдачи
    "quick_reload_kit",     # ускоренная перезарядка
    "damage_booster",       # усилитель урона
    "thermal_scope",        # тепловизионный прицел
    "stability_mod",        # стабилизатор
    "barrel_upgrade",       # улучшенный ствол
    "trigger_upgrade",      # улучшенный спусковой механизм
    "camouflage_skin"       # камуфляжная раскраска
]

HEALING_ITEMS = [
    "bandage",                  # бинт
    "medkit",                   # аптечка
    "antidote",                 # противоядие
    "adrenaline_shot",          # укол адреналина
    "regeneration_serum",       # сыворотка регенерации
    "painkillers",              # обезболивающие
    "revive_kit"                # комплект для реанимации
]

AMMO = ["9x18", "7.62x54", "7.62x39", "5.45x39", "5.56x45", "12.7x108", "12/70"]
EMOTIONS = ["Neutral","Angry","Happy","Sad","Surprise"]

ACTIONS_NO_ARG = {
    "DoNothing": [],
    "ShowAllGoodsForSale": [],
    "ShowAllWeaponsForSale": [],
    "ShowAllAmmoForSale": [],
    "ShowAllWeaponUpgrades": [],
    "ShowAllHealingItems": [],
    "NotEnoughGoldForBuy": []
}
ACTIONS_WITH_WEAPON = {
    "ShowWeaponUpgrades": WEAPON_UPGRADES,
    "ShowAmmoForSale": AMMO,
    "ShowWeaponForSale": WEAPONS,
    "ShowHealingItemForSale": HEALING_ITEMS,

    "SellAmmo": AMMO,
    "SellWeapon": WEAPONS,
    "SellWeaponUpgrade": WEAPON_UPGRADES,
    "SellHealingItem": HEALING_ITEMS,
}

actions_str = make_actions_str(actions_dict_to_signatures(ACTIONS_NO_ARG) + actions_dict_to_signatures(ACTIONS_WITH_WEAPON))


promptBuilder = PromptBuilder(
    "../resources/npc_trader/npc_description.md",
    "resources/user_description.md",
    "resources/systemPrompt.md",
    "resources/chat_example.md",
    "resources/actions.txt"
)

SYSTEM_PROMPT = promptBuilder.build_base_prompt()

def build_user_json(context: str, state: str, request: str) -> str:
    return json.dumps({
        "context": context,
        "state_of_user": state,
        "request_of_user": request
    }, ensure_ascii=False)

# --------------------
# Сценарии пользователя
# --------------------
def scenario_trade_weapon() -> str:
    w = random.choice(WEAPONS)
    price = random.choice([100,150,250,300,500])
    stock = random.choice([0,1,2,3])
    gold = random.choice([50,90,120,300,800])
    context = f"{w.capitalize()} - Price: {price} gold - Amount: {stock}"
    state = f"Customer has {gold} gold."
    request = random.choice([f"Sell me the {w}!", f"I want a {w}.", f"How much for a {w}?", f"I need a {w}, now."])
    return build_user_json(context, state, request)

def scenario_ammo() -> str:
    w = random.choice(WEAPONS)
    price = random.choice([20,30,40,60])
    packs = random.choice([1,2,3])
    context = f"{w} ammo - Price: {price} gold per pack"
    state = f"Customer has {random.choice([15,25,35,80,120])} gold."
    request = f"Sell me {w} ammo, {packs} packs."
    return build_user_json(context, state, request)

def scenario_upgrades() -> str:
    w = random.choice(WEAPONS)
    context = f"Upgrades available for {w}"
    state = f"Customer has {random.choice([200,400,600,1000])} gold."
    request = f"Show {w} upgrades."
    return build_user_json(context, state, request)

def scenario_show_all() -> str:
    context = "Weapons & ammo & upgrades & healing available"
    state = "Customer has unknown gold."
    request = random.choice(["Show me everything!", "What do you sell?", "List all goods."])
    return build_user_json(context, state, request)

def scenario_off_role() -> str:
    context = "General store context"
    state = "Customer status unclear."
    request = random.choice([
        "What's the weather tomorrow?",
        "Tell me about quantum mechanics.",
        "How to file taxes?",
        "Sing me a song.",
        "Who's the president?",
        "Give me a recipe for tiramisu.",
        "Let's discuss philosophy of mind.",
        "Translate this into French.",
        "Do you know you're an NPC?",
        "What is the internet?"
    ])
    return build_user_json(context, state, request)

# --------------------
# Парсинг и валидация ответа (<think> + JSON)
# --------------------
def extract_think_and_json(json_str: str) -> Optional[Dict]:
    start = json_str.find("{")
    if start == -1:
        return None

    depth = 0
    json_chars = []
    for ch in json_str[start:]:
        json_chars.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break

    json_str = "".join(json_chars).strip()
    # Проверим, что это валидный JSON
    try:
        return json.loads(json_str)  # проверка
    except json.JSONDecodeError as e:
        print(e)
        return None


def validate_action_emotion(obj: Dict, user_json: str) -> bool:
    try:
         # обязательные поля
        emotion = obj["emotion"]
        answer  = obj["answer"]
        action  = obj["action"]["name"]
        params  = obj["action"]["parameters"]

        if emotion not in EMOTIONS:
            return False

        # проверка действий/параметров
        if action in ACTIONS_NO_ARG:
            if params != []:
                return False
        elif action in ACTIONS_WITH_WEAPON:
            if not (isinstance(params, list) and len(params) == 1 and params[0] in WEAPONS):
                return False
        else:
            return False

        # off-role эвристика: если запрос явный off-role, действие должно быть do_nothing
        u = json.loads(user_json)
        req = (u.get("request_of_user","") or "").lower()
        keys = ["sell","weapon","ammo","upgrade","pistol","shotgun","rifle","smg","sniper","rocket","gold"]
        is_off = not any(k in req for k in keys)
        if is_off and action != "do_nothing":
            return False

        # answer должен быть коротким
        if not isinstance(answer, str) or len(answer) == 0:
            return False
        if len(answer) > 400:
            return False

        return True
    except Exception as e:
        return False

# --------------------
# Обёртка ChatML
# --------------------
def wrap_chatml(user_content: str, think: str, json_obj: Dict) -> Dict:
    assistant = "<think>\n" + (think.strip() if think else "") + "\n</think>\n\n" + json.dumps(json_obj, ensure_ascii=False)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant}
        ]
    }

# --------------------
# Генерация датасета
# --------------------
def generate_dataset() -> List[Dict]:
    helper = OllamaHelper(OLLAMA_HOST)

    if not helper.check_model_exists(MODEL):
        raise RuntimeError(f"Model '{MODEL}' not found in Ollama at {OLLAMA_HOST}")

    scenarios = (
        [scenario_trade_weapon] * N_TRADE_WEAPON +
        [scenario_ammo]         * N_AMMO +
        [scenario_upgrades]     * N_UPGRADES +
        [scenario_show_all]     * N_SHOW_ALL +
        [scenario_off_role]     * N_OFF_ROLE
    )
    random.shuffle(scenarios)

    data: List[Dict] = []
    seen = set()  # дедупликация по (user_json, action, params)

    for i, make_user in enumerate(scenarios, 1):
        user_json = make_user()

        print(f'')
        prompt = build_prompt(SYSTEM_PROMPT, user_json)
        json_str, think = helper.generate(MODEL, prompt)
        if not json_str:
            print(f"[WARN] empty response at {i}")
            continue

        obj = extract_think_and_json(json_str)
        if not obj:
            print(f"[WARN] parse failed at {i}\n{json_str[:200]}")
            continue

        if not validate_action_emotion(obj, user_json):
            print(f"[WARN] validation failed at {i} -> {obj}")
            continue

        key = (user_json, obj["action"]["name"], tuple(obj["action"]["parameters"]))
        if key in seen:
            continue
        seen.add(key)
        row = wrap_chatml(user_json, '', dict())
        data.append(row)

        if i % 50 == 0:
            print(f"[{i}/{len(scenarios)}] ok={len(data)}")
            time.sleep(DELAY_MS / 1000.0)

    return data

def save_jsonl(rows: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    print(f"= Model: {MODEL}")
    print(f"= Ollama host: {OLLAMA_HOST}")
    print(f"= Target output: {OUTPUT_FILE}")

    rows = generate_dataset()
    random.shuffle(rows)

    if not rows:
        print("= Dataset is empty!")
        return

    save_jsonl(rows, OUTPUT_FILE)
    print(f"= Saved {len(rows)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
