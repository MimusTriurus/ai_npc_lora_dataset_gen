import json
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class Action:
    name: str
    parameters: List[str]

    @staticmethod
    def parse_action(action_str: str):
        name = action_str.split("(", 1)[0].strip()
        match = re.search(r"\((.*)\)", action_str)
        if not match:
            return Action(name=name, parameters=[])
        inside = match.group(1).strip()
        if not inside:
            return Action(name=name, parameters=[])
        params = [p.strip() for p in inside.split(",")]
        #params = [p.strip("<>") for p in raw_params]
        return Action(name=name, parameters=params)

    def __eq__(self, other):
        return self.name == other.name and self.parameters == other.parameters

@dataclass
class Question:
    template: str
    action: str
    motivation: str
    context: str

    def __hash__(self):
        return hash(self.template)

    def __eq__(self, other):
        return isinstance(other, Question) and self.template == other.template

    def __dict__(self):
        return {
            "template": self.template,
            "action": self.action,
            "motivation": self.motivation,
            "context": self.context,
        }

@dataclass
class PlayerRole:
    name: str
    description: str
    speech_style: str

    def __str__(self):
        return f'Name: {self.name}\nDescription: {self.description}\nSpeech style: {self.speech_style}'