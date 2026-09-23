import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from enum import Enum, auto
from typing import Any


class GameState(Enum):
    CHOOSE_PILE = auto()
    CHOOSE_CHIP = auto()
    CHOOSE_NEXT_PLAYER = auto()
    ELIMINATE_CHIP = auto()


class Player:
    def __init__(
        self,
        letter: str,
        max_chips: int,
        player_letters: list[str],
    ) -> None:
        self.letter: str = letter
        self.chips: dict[str, int] = {
            l: max_chips if l == letter else 0 for l in player_letters
        }
        self.dead_chips: int = 0
        self.eliminated: bool = False
        self.reward: float = 0.0

    def has_only_one_chip_type(self) -> bool:
        return sum(1 for count in self.chips.values() if count > 0) == 1

    def get_only_chip_type(self) -> str:
        return next(letter for letter, count in self.chips.items() if count > 0)

    def count_total_chips(self) -> int:
        return sum(self.chips.values())


class ChipGameEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        # Constants
        self.NUM_PLAYERS: int = 4
        self.PLAYER_LETTERS: list[str] = ["A", "B", "C", "D"]
        self.MAX_CHIPS: int = 5
        self.NUM_PILES: int = 6

        # Gym variables
        self.done: bool = False

        self.action_space: spaces.Space = spaces.Discrete(
            self.NUM_PILES + self.NUM_PLAYERS
        )

        MAX_PILE_SIZE: int = self.NUM_PLAYERS * self.MAX_CHIPS

        TOTAL_OBSERVATION_SIZE: int = (
            self.NUM_PILES * self.NUM_PLAYERS * MAX_PILE_SIZE  # Board state
            + self.NUM_PLAYERS * self.NUM_PLAYERS  # Player chips
            + self.NUM_PLAYERS  # Dead chips
            + self.NUM_PLAYERS  # Current player
            + len(GameState)  # Game state
            + 1  # Steps
        )

        self.observation_space: spaces.Space = spaces.Box(
            low=0,
            high=max(self.MAX_CHIPS, MAX_PILE_SIZE),
            shape=(TOTAL_OBSERVATION_SIZE,),
            dtype=np.int32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.players: list[Player] = [
            self._create_player(letter) for letter in self.PLAYER_LETTERS
        ]
        self.current_player_index: int = random.randint(0, self.NUM_PLAYERS - 1)
        self.piles: list[list[str]] = [[] for _ in range(self.NUM_PILES)]
        self.state: GameState = GameState.CHOOSE_PILE
        self.last_played_pile: int | None = None
        self.turn_history: list[Player] = [self.players[0]]
        self.eligible_next_players: list[Player] = []
        self.done: bool = False
        self.steps_num: int = 0

        observation: np.ndarray = self._get_obs()
        info: dict[str, Any] = {}

        return observation, info

    @staticmethod
    def reward_fn(reward: float, chips: int, steps: int) -> float:
        beta: int = 50
        alpha: float = 15 / (chips * beta)

        return min(reward, reward / (alpha * steps))

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, dict[str, float], bool, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        winner: Player | None = None
        step_log: list[str] = []
        step_rewards: dict[str, float] = {player.letter: 0.0 for player in self.players}

        if self.done:
            step_log.append("Game is already over. No more actions can be taken.\n")
            observation: np.ndarray = self._get_obs()
            info["winner"] = winner
            info["log"] = step_log
            return observation, step_rewards, self.done, False, info

        self.steps_num += 1

        # Choose Pile
        if self.state == GameState.CHOOSE_PILE:
            if 0 <= action < self.NUM_PILES:
                self.last_played_pile = action

                if self._current_player().has_only_one_chip_type():
                    if self._play_chip(
                        self.last_played_pile,
                        self._current_player().get_only_chip_type(),
                    ):
                        reward: float = self.reward_fn(
                            5.0, self.MAX_CHIPS, self.steps_num
                        )
                        self._current_player().reward += reward
                        step_rewards[self._current_player().letter] += reward
                    else:
                        self._current_player().reward -= 5.0
                        step_rewards[self._current_player().letter] -= 5.0
                else:
                    self.state = GameState.CHOOSE_CHIP
            else:
                self._current_player().reward -= 5.0
                step_rewards[self._current_player().letter] -= 5.0

        # Choose Chip
        elif self.state == GameState.CHOOSE_CHIP:
            if self.NUM_PILES <= action < self.NUM_PILES + self.NUM_PLAYERS:
                chip_letter: str = self.PLAYER_LETTERS[action - self.NUM_PILES]

                if self._play_chip(self.last_played_pile, chip_letter):
                    reward: float = self.reward_fn(5.0, self.MAX_CHIPS, self.steps_num)
                    self._current_player().reward += reward
                    step_rewards[self._current_player().letter] += reward
                else:
                    self._current_player().reward -= 5.0
                    step_rewards[self._current_player().letter] -= 5.0
            else:
                self._current_player().reward -= 5.0
                step_rewards[self._current_player().letter] -= 5.0

        # Choose Next Player
        elif self.state == GameState.CHOOSE_NEXT_PLAYER:
            if self.NUM_PILES <= action < self.NUM_PILES + self.NUM_PLAYERS:
                next_player_letter: str = self.PLAYER_LETTERS[action - self.NUM_PILES]

                if self._choose_next_player(next_player_letter):
                    reward: float = self.reward_fn(5.0, self.MAX_CHIPS, self.steps_num)
                    self._current_player().reward += reward
                    step_rewards[self._current_player().letter] += reward
                    self.state = GameState.CHOOSE_PILE
                else:
                    self._current_player().reward -= 5.0
                    step_rewards[self._current_player().letter] -= 5.0
            else:
                self._current_player().reward -= 5.0
                step_rewards[self._current_player().letter] -= 5.0

        # Eliminate Chip
        elif self.state == GameState.ELIMINATE_CHIP:
            if self.NUM_PILES <= action < self.NUM_PILES + self.NUM_PLAYERS:
                chip_to_eliminate: str = self.PLAYER_LETTERS[action - self.NUM_PILES]

                if self._eliminate_chip(chip_to_eliminate):
                    reward: float = self.reward_fn(5.0, self.MAX_CHIPS, self.steps_num)
                    self._current_player().reward += reward
                    step_rewards[self._current_player().letter] += reward
                    self.state = GameState.CHOOSE_PILE
                else:
                    self._current_player().reward -= 5.0
                    step_rewards[self._current_player().letter] -= 5.0
            else:
                self._current_player().reward -= 5.0
                step_rewards[self._current_player().letter] -= 5.0

        else:
            self._current_player().reward -= 5.0
            step_rewards[self._current_player().letter] -= 5.0

        # Logging
        step_log.append(f"Current Player: {self._current_player().letter}\n")
        step_log.append(f"Current State: {self.state.name}\n")
        step_log.append(f"Action: {action}\n")
        step_log.append(f"Last Played Pile: {self.last_played_pile}\n")
        step_log.append(
            f"Turn History: {[player.letter for player in self.turn_history]}\n"
        )
        step_log.append(f"Active Players: {self._count_active_players()}\n")
        step_log.append(f"Piles: {self.piles}\n")

        for player in self.players:
            step_log.append(f"{player.letter}:\n")

            for letter, count in player.chips.items():
                step_log.append(f"  {letter}: {count}\n")

            step_log.append(f"Dead Chips: {player.dead_chips}\n")
            step_log.append("\n")

        # Update player rewards and track step rewards
        for player in self.players:
            step_rewards[player.letter] = player.reward
            player.reward = 0.0

        # Check game-over condition
        if self.is_game_over():
            self.done = True

            step_log.append("Game Over Condition Met\n")

            winner = next(
                (player for player in self.players if not player.eliminated),
                None,
            )

            if winner is not None:
                winner.reward += 15.0
                step_rewards[winner.letter] += 15.0
                step_log.append(f"Winner ({winner.letter}) reward: 15\n")

        observation: np.ndarray = self._get_obs()

        info["winner"] = winner
        info["log"] = step_log

        return observation, step_rewards, self.done, False, info


    def _get_obs(self) -> np.ndarray:
        max_pile_size: int = self.NUM_PLAYERS * self.MAX_CHIPS

        total_observation_size: int = (
            self.NUM_PILES * self.NUM_PLAYERS * max_pile_size  # Board state
            + self.NUM_PLAYERS * self.NUM_PLAYERS  # Player chips
            + self.NUM_PLAYERS  # Dead chips
            + self.NUM_PLAYERS  # Current player
            + len(GameState)  # Game state
            + 1  # Steps
        )

        observation: np.ndarray = np.zeros(
            total_observation_size,
            dtype=np.int32,
        )

        # Board state
        for i, pile in enumerate(self.piles):
            for j, chip in enumerate(pile):
                if j < max_pile_size:
                    observation[
                        i * self.NUM_PLAYERS * max_pile_size
                        + self.PLAYER_LETTERS.index(chip) * max_pile_size
                        + j
                    ] = 1

        # Player chips
        offset: int = self.NUM_PILES * self.NUM_PLAYERS * max_pile_size

        for i, player in enumerate(self.players):
            for j, letter in enumerate(self.PLAYER_LETTERS):
                observation[offset + i * self.NUM_PLAYERS + j] = player.chips[letter]

        # Dead chips
        offset += self.NUM_PLAYERS * self.NUM_PLAYERS

        for i, player in enumerate(self.players):
            observation[offset + i] = player.dead_chips

        # Current player
        offset += self.NUM_PLAYERS
        observation[offset + self.current_player_index] = 1

        # Game state
        offset += self.NUM_PLAYERS

        state_index: int = list(GameState).index(self.state)
        observation[offset + state_index] = 1

        # Number of steps
        offset += len(GameState)
        observation[offset] = self.steps_num

        return observation
    
    def _create_player(self, letter: str) -> Player:
        return Player(
            letter,
            self.MAX_CHIPS,
            self.PLAYER_LETTERS,
        )

    def _current_player(self) -> Player:
        return self.players[self.current_player_index]

    def _play_chip(self, pile: int, chip_letter: str) -> bool:
        if (
            0 <= pile < self.NUM_PILES
            and self._current_player().chips[chip_letter] > 0
        ):
            self.piles[pile].append(chip_letter)
            self._current_player().chips[chip_letter] -= 1
            self.last_played_pile = pile

            if self._check_capture(pile):
                benefitting_player: Player | None = next(
                    (
                        player
                        for player in self.players
                        if player.letter == chip_letter
                    ),
                    None,
                )

                # Capturing player is still active
                if benefitting_player is not None and not benefitting_player.eliminated:
                    self.current_player_index = self.players.index(
                        benefitting_player
                    )
                    self.state = GameState.ELIMINATE_CHIP

                # Capturing player has already been eliminated:
                # dead-zone the entire pile
                else:
                    for chip in self.piles[self.last_played_pile]:
                        player: Player | None = next(
                            (
                                player
                                for player in self.players
                                if player.letter == chip
                            ),
                            None,
                        )

                        if player is not None:
                            player.dead_chips += 1

                    self.piles[self.last_played_pile] = []

                    if self._current_player().count_total_chips() == 0:
                        self._check_player_elimination()

                    self.state = GameState.CHOOSE_PILE

            else:
                self._determine_next_players()

            return True

        return False

    def _check_capture(self, pile: int) -> bool:
        played_pile: list[str] = self.piles[pile]

        return (
            len(played_pile) > 1
            and played_pile[-1] == played_pile[-2]
        )

    def _determine_next_players(self) -> None:
        played_pile: list[str] = self.piles[self.last_played_pile]
        active_players: list[Player] = [
            p for p in self.players if not p.eliminated
        ]
        active_player_letters: list[str] = [
            p.letter for p in active_players
        ]
        colors_in_pile: list[str] = [
            color
            for color in set(played_pile)
            if not next(
                player.eliminated
                for player in self.players
                if player.letter == color
            )
        ]

        if sorted(colors_in_pile) == sorted(active_player_letters):
            last_appearances: dict[str, int] = {
                color: len(played_pile) - 1 - played_pile[::-1].index(color)
                for color in colors_in_pile
            }

            next_player_letter: str = min(
                last_appearances,
                key=last_appearances.get,
            )

            self._set_next_player(
                next(
                    p
                    for p in active_players
                    if p.letter == next_player_letter
                )
            )
            self.state = GameState.CHOOSE_PILE

        else:
            self.eligible_next_players = [
                p
                for p in active_players
                if p.letter not in colors_in_pile
            ]

            if len(self.eligible_next_players) == 1:
                self._set_next_player(self.eligible_next_players[0])
                self.state = GameState.CHOOSE_PILE
            else:
                self.state = GameState.CHOOSE_NEXT_PLAYER

    def _choose_next_player(self, letter: str) -> bool:
        if any(
            p.letter == letter
            for p in self.eligible_next_players
        ):
            self._set_next_player(
                next(
                    p
                    for p in self.players
                    if p.letter == letter
                )
            )
            return True

        return False

    def _set_next_player(self, player: Player) -> None:
        self.current_player_index = self.players.index(player)
        self.turn_history.append(player)
        self._check_player_elimination()

    def _check_player_elimination(self) -> bool:
        while (
            sum(self._current_player().chips.values()) == 0
            and not self._current_player().eliminated
        ):
            if self.turn_history and self._count_active_players() > 1:
                self._current_player().eliminated = True

                self.turn_history = [
                    player
                    for player in self.turn_history
                    if player != self._current_player()
                ]

                if len(self.turn_history) > 0:
                    self.current_player_index = self.players.index(
                        self.turn_history[-1]
                    )
                else:
                    self.current_player_index = next(
                        i
                        for i, player in enumerate(self.players)
                        if not player.eliminated
                    )

            else:
                self.done = True
                self.step(0)
                return True

        return False

    def _eliminate_chip(self, chip_letter: str) -> bool:
        captured_pile: list[str] = self.piles[self.last_played_pile]

        # Check if the chip being eliminated is actually in the pile
        if chip_letter in captured_pile:
            player: Player | None = next(
                (p for p in self.players if p.letter == chip_letter),
                None,
            )

            # Increment eliminated chip count
            player.dead_chips += 1

            # Add all chips in the captured pile to the current player's pocket
            for chip in captured_pile:
                self._current_player().chips[chip] += 1

            # Remove the eliminated chip
            self._current_player().chips[chip_letter] -= 1

            # Empty the captured pile and reset state
            self.piles[self.last_played_pile] = []
            self.state = GameState.CHOOSE_PILE

            return True

        # The chip being eliminated is not in the pile
        return False

    def _count_active_players(self) -> int:
        return sum(
            1
            for player in self.players
            if not player.eliminated
        )

    def is_game_over(self) -> bool:
        return self._count_active_players() <= 1

    def render(self) -> None:
        pass

    def close(self) -> None:
        pygame.quit()

