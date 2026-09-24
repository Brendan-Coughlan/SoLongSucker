from abc import ABC, abstractmethod
from chip_game_env import ChipGameEnv

class Agent(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def choose_action(self, env: ChipGameEnv) -> int:
        pass