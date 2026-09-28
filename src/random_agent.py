import random
from base_agent import BaseAgent
from chip_game_env import ChipGameEnv, GameState


class RandomAgent(BaseAgent):
    def __init__(self, name, state_size, action_size, **kwargs):
        super().__init__(name, state_size, action_size, **kwargs)

    def choose_action(self, env: ChipGameEnv) -> int:
        # Choose a random pile when the game requires a pile selection.
        if env.state == GameState.CHOOSE_PILE:
            return random.randint(0, env.NUM_PILES - 1)

        # Otherwise, randomly choose one of the player-based actions.
        return random.randint(
            env.NUM_PILES,
            env.action_space.n - 1,
        )