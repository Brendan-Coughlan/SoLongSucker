from agent import Agent
from chip_game_env import ChipGameEnv, GameState


class DDQNAgent(Agent):
    def choose_action(self, env: ChipGameEnv) -> int:
        return 0