from jinja2 import Environment, ext
import random
from typing import List, Tuple, Dict
import re

# region Jinja template configuration
def rand_range(a: int, b: int) -> int:
    return random.randint(a, b)

def join(lst: list) -> str:
    return ", ".join(lst)

class IterableExpansionExtension(ext.Extension):
    pattern = re.compile(r"\{\{\s*(\w+)\[\]\s*\}\}")

    def preprocess(self, source, name, filename=None):
        variables = self.pattern.findall(source)

        if not variables:
            return source

        seen = []
        for v in variables:
            if v not in seen:
                seen.append(v)

        for var in seen:
            source = re.sub(
                rf"\{{\{{\s*{var}\[\]\s*\}}\}}",
                f"{{{{ {var}_it }}}}",
                source
            )

        loop_open = [
            f"{{% for {var}_it in {var} %}}"
            for var in seen
        ]
        loop_close = ["{% endfor %}" for _ in seen]

        wrapped = "\n".join(loop_open) + "\n"
        wrapped += source.strip() + "\n"
        wrapped += "\n".join(reversed(loop_close))

        return wrapped

def make_jinja_environment() -> Environment:
    env_ = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[IterableExpansionExtension]
    )
    env_.globals["rand_range"] = rand_range
    env_.globals["join"] = join
    return env_

def build_action_template_params(data: dict) -> dict:
    parameters = data['Parameters']
    result = {}
    for parameter_name, parameter_values in parameters.items():
        result[parameter_name] = parameter_values
    result['Parameters'] = parameters
    return result

def render_template(template_str: str, context: dict) -> List[str]:
    template = env.from_string(template_str)
    result = []
    rendered_requests = template.render(**context).split('\n')
    for r in rendered_requests:
        if not r:
            continue
        result.append(r)
    return result

env = make_jinja_environment()
# endregion