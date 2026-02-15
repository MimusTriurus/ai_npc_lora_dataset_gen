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
    parameters: ActionParameters


@dataclass
class NPCValidAction:
    emotion: str
    answer: str
    think: str
    action: Action


@dataclass
class Root:
    usr_request: UserRequest
    npc_valid_action: NPCValidAction