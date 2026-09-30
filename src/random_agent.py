import random
from chip_game_env import ChipGameEnv, GameState

class RandomAgent:
    """Agent that selects uniformly random actions."""

    def __init__(self, name: str) -> None:
        self.name = name

    def choose_action(self, env: ChipGameEnv) -> int:
        """Choose a random action based on the current game state."""

        if env.state == GameState.CHOOSE_PILE:
            return random.randint(0, env.NUM_PILES - 1)

        return random.randint(
            env.NUM_PILES,
            env.action_space.n - 1
        )