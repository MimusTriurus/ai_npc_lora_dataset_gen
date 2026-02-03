import os
import re
from dataclasses import asdict
import json
from pathlib import Path
from dataclasses import is_dataclass
from typing import List, Type, TypeVar, Dict, Iterable, Set, Any
from common.data_classes import Action, Question


def camel_to_snake(name: str) -> str:
    """Превращает CamelCase / PascalCase в snake_case."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i-1].isupper()):
            out.append('_')
        out.append(ch.lower())
    return ''.join(out)

def unique_stable(seq: Iterable[str]) -> List[str]:
    """Удаляет дубликаты, сохраняя порядок (stable de-dup)."""
    seen = set()
    res = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

def actions_dict_to_signatures(
        actions: Dict[str, List[str]],
        sep: str = " | ",
        with_spaces_in_parens: bool = True
) -> List[str]:
    result = []
    for action_name, params in actions.items():
        # удалим дубликаты, сохраняя порядок
        params_unique = unique_stable(params)

        if params_unique:
            joined = sep.join(params_unique)
            if with_spaces_in_parens:
                signature = f"{action_name}( {joined} )"
            else:
                signature = f"{action_name}({joined})"
        else:
            # если параметров нет, вернём пустые скобки
            signature = f"{action_name}()"

        result.append(signature)

    return result

def make_actions_str(actions: List[str]) -> str:
    actions_str = '- ' + '\n- '.join(actions)
    return actions_str

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class PromptBuilder:
    def __init__(
            self,
            npc_desc_file: str,
            user_desc_file: str,
            system_prompt_file: str,
            chat_example_file: str,
            actions_file: str
    ):
        self.npc_desc_file = npc_desc_file
        self.user_desc_file = user_desc_file
        self.system_prompt_file = system_prompt_file
        self.chat_example_file = chat_example_file
        self.actions_file = actions_file

        self.npc_desc = ""
        self.user_desc = ""
        self.system_prompt = ""
        self.chat_example = ""
        self.actions_text = ""

        self.actions = {}

    def read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_actions_to_dict(self) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}

        pattern = re.compile(r'^\s*-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*?)\))?\s*$', re.UNICODE)

        for raw_line in self.actions_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            m = pattern.match(line)
            if not m:
                # строка не соответствует ожидаемому формату — пропускаем
                continue

            action_name = m.group(1)
            params_str = m.group(2)

            if params_str is None or params_str.strip() == "":
                # нет параметров
                result[action_name] = []
            else:
                # параметры разделены '|' (возможны пробелы вокруг)
                params = [p.strip() for p in params_str.split('|') if p.strip()]
                result[action_name] = params

        return result

    def make_system_prompt(
            self,
            system_prompt: str,
            npc_description: str,
            user_description: str,
            actions: str,
            chat_example: str
    ) -> str:
        # Подставляем плейсхолдеры
        system_prompt = re.sub(r"<npc_description></npc_description>", npc_description, system_prompt)
        system_prompt = re.sub(r"<user_description></user_description>", user_description, system_prompt)
        system_prompt = re.sub(r"<actions></actions>", actions, system_prompt)
        system_prompt = re.sub(r"<chat_example></chat_example>", chat_example, system_prompt)
        return system_prompt

    def build_base_prompt(self) -> str:
        self.npc_desc = self.read_file(self.npc_desc_file)
        self.user_desc = self.read_file(self.user_desc_file)
        self.system_prompt = self.read_file(self.system_prompt_file)
        self.chat_example = self.read_file(self.chat_example_file)
        self.actions_text = self.read_file(self.actions_file)

        prompt = self.make_system_prompt(
            self.system_prompt,
            self.npc_desc,
            self.user_desc,
            self.actions_text,
            self.chat_example
        )

        self.actions = self.parse_actions_to_dict()

        return prompt

    def get_actions(self) -> List[Action]:
        actions: List[Action] = []
        with open(self.actions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("-"):
                    continue

                if line.startswith("- "):
                    line = line[2:]
                else:
                    line = line[1:]

                open_paren = line.find("(")
                close_paren = line.find(")")

                if open_paren == -1 or close_paren == -1:
                    name = line.strip()
                    params = []
                else:
                    name = line[:open_paren].strip()
                    params_str = line[open_paren + 1:close_paren]
                    params = [p.strip() for p in params_str.split(",") if p.strip()]

                actions.append(Action(name=name, parameters=params))

def save_questions_to_jsonl(
    questions: Set[Question],
    output_file: str,
    append: bool = False
) -> None:
    path = Path(output_file)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for q in questions:
            record = {
                "base_template": q.template,
                "action": q.action,
                "motivation": q.motivation,
                "context": q.context,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def save_dataclass_records_to_jsonl(
    records: List[Any],
    output_file: str,
    append: bool = False
) -> None:
    path = Path(output_file)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            json_str = json.dumps(asdict(r))
            f.write(json_str + "\n")

def save_dict_records_to_jsonl(
    records: List[dict],
    output_file: str,
    append: bool = False
) -> None:
    path = Path(output_file)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in records:
            json_str = json.dumps(r)
            f.write(json_str + "\n")

T = TypeVar("T")

def load_jsonl_to_dataclasses(
    input_file: str,
    cls: Type[T]
) -> List[T]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    path = Path(input_file)
    records: List[T] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            obj = cls(**data)
            records.append(obj)

    return records

def extract_angle_bracket_substrings(text: str):
    return re.findall(r"<[^>]*>", text)

def list_files(folder_path: str) -> list[str]:
    return [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, name))
    ]

def calculate_dataset_params(
    dataset_size=4000,
    actions=4,
    params=6,
    roles_min=1,
    roles_max=200,
    queries_min=1,
    queries_max=200,
    tolerance=0.05
):
    target = dataset_size
    base = actions * params
    ideal = target / base

    solutions = []

    for r in range(roles_min, roles_max + 1):
        for q in range(queries_min, queries_max + 1):
            produced = base * r * q
            error = abs(produced - target) / target

            if error <= tolerance:
                solutions.append({
                    "roles": r,
                    "queries": q,
                    "produced": produced,
                    "error": error
                })

    if solutions:
        solutions = sorted(solutions, key=lambda x: (x["roles"], x["produced"]), reverse=True)

    return solutions

