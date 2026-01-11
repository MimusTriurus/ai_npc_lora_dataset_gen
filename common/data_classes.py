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
class UserRequest:
    context: str
    state_of_user: str
    request_of_user: str

    def __dict__(self):
        return {
            "context": self.context,
            "state_of_user": self.state_of_user,
            "request_of_user": self.request_of_user,
        }

    def to_string(self):
        request = {
            'context': self.context,
            'state_of_user': self.state_of_user,
            'request_of_user': self.request_of_user,
        }
        result = json.dumps(request)
        return result

@dataclass
class NpcResponse:
    emotion: str
    answer: str
    think: str
    action: Action

@dataclass
class RequestResponsePair:
    user_request: UserRequest
    npc_response: NpcResponse

    def is_valid(self) -> bool:
        return self.npc_response.answer != ""