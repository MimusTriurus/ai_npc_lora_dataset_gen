from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class UserRequest:
    request: str
    usr_state: str
    npc_state: str


@dataclass
class ActionParameters:
    item: str


@dataclass
class Action:
    name: str
    parameters: dict

    def __str__(self):
        parameters_str = ""
        for key, value in self.parameters.items():
            parameters_str += f'{key}="{value}", '
        if parameters_str:
            parameters_str = parameters_str[:-2]
        return f'{self.name}({parameters_str})'


@dataclass
class NPCValidAction:
    emotion: str
    answer: str
    think: str
    action: Action


@dataclass
class PlayerRole:
    name: str
    description: str
    speech_style: str

    def __str__(self):
        return f'Name: {self.name}\nDescription: {self.description}\nSpeech style: {self.speech_style}'


@dataclass
class Root:
    usr_request: UserRequest
    npc_valid_action: NPCValidAction
    player_role: PlayerRole