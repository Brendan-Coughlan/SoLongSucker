import random
from agent import Agent
from chip_game_env import ChipGameEnv, GameState


class RandomAgent(Agent):
    def choose_action(self, env: ChipGameEnv) -> int:
        # Choose a random pile when the game requires a pile selection.
        if env.state == GameState.CHOOSE_PILE:
            return random.randint(0, env.NUM_PILES - 1)

        # Otherwise, randomly choose one of the player-based actions.
        return random.randint(
            env.NUM_PILES,
            env.action_space.n - 1,
        )