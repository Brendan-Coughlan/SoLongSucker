from chip_game_env import ChipGameEnv
from random_agent import play_game_with_multiple_agents

if __name__ == "__main__":
    env: ChipGameEnv = ChipGameEnv()

    play_game_with_multiple_agents(
        env=env,
        num_episodes=1000,
        max_steps=100,
        render=True,
    )