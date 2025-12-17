from ollama_helper import *

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == '__main__':
    system_prompt = read_file('resources/system_prompt_gen_questions.md')
    base_template = "Show me the <weapon> you're selling."
    user_state = 'Wounded brave soldier'
    template_request = f"Player's role: {user_state}. Create another template for : {base_template}"
    templates = set()
    templates.add(base_template)
    system_prompt += f'\n* {base_template}'

    prompt = build_prompt(system_prompt, template_request)
    helper = OllamaHelper(OLLAMA_HOST)
    MAX = 10
    counter = MAX
    while counter:
        new_template, think = helper.generate(MODEL, prompt)
        if new_template not in templates:
            print(f'==> [{MAX - counter}] {new_template}')
            system_prompt += f'\n* {new_template}'
            counter -= 1
            templates.add(new_template)