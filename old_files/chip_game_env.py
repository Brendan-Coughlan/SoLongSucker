import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class ChipGameEnv(gym.Env):
    def __init__(self):
        super(ChipGameEnv, self).__init__()

        self.num_players = 4
        self.player_letters = ["A", "B", "C", "D"]
        self.max_chips = 5
        self.num_rows = 6
        self.done = False

        # Action and observation spaces
        self.action_space = spaces.Discrete(self.num_rows + 4)
        # Calculate the total observation space size
        max_pile_size = self.num_players * self.max_chips
        total_obs_size = (
            self.num_rows * self.num_players * max_pile_size  # Board state
            + self.num_players * self.num_players  # Player chips
            + self.num_players  # Dead chips
            + self.num_players  # Current player
            + 4  # Game state (assuming 4 possible states)
            + 1  # Steps
        )

        self.observation_space = spaces.Box(
            low=0,
            high=max(self.max_chips, max_pile_size),
            shape=(total_obs_size,),
            dtype=np.int32,
        )

        # Initialize Pygame for rendering
        pygame.init()
        print("Pygame initialized")
        # self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Chip Game")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.players = [self._create_player(letter) for letter in self.player_letters]
        self.current_player_index = random.randint(0, self.num_players - 1)
        self.rows = [[] for _ in range(self.num_rows)]
        self.state = "choose_pile"
        self.last_played_row = None
        self.turn_history = [self.players[0]]
        self.eligible_next_players = []
        self.done = False
        self.steps_num = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    @staticmethod
    def reward_fn(reward, chips, steps):
        beta = 50
        alpha = 15/(chips*beta)
        return min(reward, reward/(alpha*steps))


    def step(self, action):
        info = {}
        winner = None
        step_log = []
        step_rewards = {player.letter: 0 for player in self.players}  # Track rewards for each player

        if self.done:
            step_log.append("Game is already over. No more actions can be taken.\n")
            observation = self._get_obs()
            info['winner'] = winner
            info['log'] = step_log
            return observation, 0, self.done, False, info

        self.steps_num += 1

        if self.state == "choose_pile":
            if 0 <= action < self.num_rows:
                self.last_played_row = action
                if self._current_player().has_only_one_chip_type():
                    if self._play_chip(self.last_played_row, self._current_player().get_only_chip_type()):
                        self._current_player().reward += self.reward_fn(5, self.max_chips, self.steps_num)  # Reward for successful chip placement
                        step_rewards[self._current_player().letter] += self.reward_fn(5, self.max_chips, self.steps_num)
                    else:
                        self._current_player().reward -= 5  # Invalid action penalty
                        step_rewards[self._current_player().letter] -= 5
                else:
                    self.state = "choose_chip"
            else:
                self._current_player().reward -= 5  # Invalid action penalty
                step_rewards[self._current_player().letter] -= 5

        elif self.state == "choose_chip":
            if self.num_rows <= action < self.num_rows + 4:
                chip_letter = self.player_letters[action - self.num_rows]
                if self._play_chip(self.last_played_row, chip_letter):
                    self._current_player().reward += self.reward_fn(5, self.max_chips, self.steps_num)  # Reward for successful chip placement
                    step_rewards[self._current_player().letter] += self.reward_fn(5, self.max_chips, self.steps_num)
                else:
                    self._current_player().reward -= 5  # Invalid action penalty
                    step_rewards[self._current_player().letter] -= 5
            else:
                self._current_player().reward -= 5  # Invalid action penalty
                step_rewards[self._current_player().letter] -= 5

        elif self.state == "choose_next_player":
            if self.num_rows <= action < self.num_rows + 4:
                next_player_letter = self.player_letters[action - self.num_rows]
                if self._choose_next_player(next_player_letter):
                    self._current_player().reward += self.reward_fn(5, self.max_chips, self.steps_num)  # Small reward for choosing next player
                    step_rewards[self._current_player().letter] += self.reward_fn(5, self.max_chips, self.steps_num)
                    self.state = "choose_pile"
                else:
                    self._current_player().reward -= 5  # Invalid action penalty
                    step_rewards[self._current_player().letter] -= 5
            else:
                self._current_player().reward -= 5  # Invalid action penalty
                step_rewards[self._current_player().letter] -= 5

        elif self.state == "eliminate_chip":
            if self.num_rows <= action < self.num_rows + 4:
                chip_to_eliminate = self.player_letters[action - self.num_rows]
                if self._eliminate_chip(chip_to_eliminate):
                    self._current_player().reward += self.reward_fn(5, self.max_chips, self.steps_num)  # Reward for successful chip elimination
                    step_rewards[self._current_player().letter] += self.reward_fn(5, self.max_chips, self.steps_num)
                    self.state = "choose_pile"
                else:
                    self._current_player().reward -= 5  # Invalid action penalty
                    step_rewards[self._current_player().letter] -= 5
            else:
                self._current_player().reward -= 5  # Invalid action penalty
                step_rewards[self._current_player().letter] -= 5
        else:
            self._current_player().reward -= 5  # Invalid action penalty
            step_rewards[self._current_player().letter] -= 5



        # step_log.append(f"Current Player: {self._current_player().letter}\n")
        # step_log.append(f"Current State: {self.state}\n")
        # step_log.append(f"Action: {action}\n")
        # step_log.append(f"Last Played Row: {self.last_played_row}\n")
        # step_log.append(f"Turn History: {[player.letter for player in self.turn_history]}\n")
        # step_log.append(f"Active Players: {self._count_active_players()}\n")
        # step_log.append(f"Rows: {self.rows}\n")

        # for player in self.players:
        #     step_log.append(f"{player.letter}:\n")
        #     for letter, count in player.chips.items():
        #         step_log.append(f"  {letter}: {count}\n")
        #     step_log.append(f"Dead Chips: {player.dead_chips}\n")
        #     step_log.append("\n")

        # Update player rewards and track step rewards
        for player in self.players:
            step_rewards[player.letter] = player.reward
            player.reward = 0  # Reset for next step



        if self.is_game_over():
            self.done = True
            steps_num = 0
            step_log.append("Game Over Condition Met\n")
            winner = next((player for player in self.players if not player.eliminated), None)
            if winner:
                winner.reward += 15
                step_rewards[winner.letter] += 15
                step_log.append(f"Winner ({winner.letter}) reward: 10\n")

        observation = self._get_obs()
        info['winner'] = winner
        info['log'] = step_log

        return observation, step_rewards, self.done, False, info




    def _get_obs(self):
        max_pile_size = self.num_players * self.max_chips
        total_obs_size = (
            self.num_rows * self.num_players * max_pile_size +  # Board state
            self.num_players * self.num_players +  # Player chips
            self.num_players +  # Dead chips
            self.num_players +  # Current player
            4 +  # Game state (assuming 4 possible states)
            1  # Steps
        )

        obs = np.zeros(total_obs_size, dtype=np.int32)

        # Board state
        for i, row in enumerate(self.rows):
            for j, chip in enumerate(row):
                if j < max_pile_size:
                    obs[i * self.num_players * max_pile_size + self.player_letters.index(chip) * max_pile_size + j] = 1

        # Player chips
        offset = self.num_rows * self.num_players * max_pile_size
        for i, player in enumerate(self.players):
            for j, letter in enumerate(self.player_letters):
                obs[offset + i * self.num_players + j] = player.chips[letter]

        # Dead chips
        offset += self.num_players * self.num_players
        for i, player in enumerate(self.players):
            obs[offset + i] = player.dead_chips

        # Current player
        offset += self.num_players
        obs[offset + self.current_player_index] = 1

        # Game state
        offset += self.num_players
        state_index = ["choose_pile", "choose_chip", "choose_next_player", "eliminate_chip"].index(self.state)
        obs[offset + state_index] = 1

        # The number of steps that have happened in the game
        offset += 4
        obs[offset] = self.steps_num

        return obs

    def _create_player(self, letter):
        return self.Player(letter, self.max_chips, self.player_letters)

    def _current_player(self):
        return self.players[self.current_player_index]

    def _play_chip(self, row, chip_letter):
        if 0 <= row < self.num_rows and self._current_player().chips[chip_letter] > 0:
            self.rows[row].append(chip_letter)
            self._current_player().chips[chip_letter] -= 1
            self.last_played_row = row
            if self._check_capture(row):
                benefitting_player = next((p for p in self.players if p.letter == chip_letter), None)
                if benefitting_player and not benefitting_player.eliminated: #do this only if capturing colored player is not eliminated otherwise entire pile is deadzoned
                    self.current_player_index = self.players.index(next(p for p in self.players if p.letter == chip_letter) )
                    self.state = "eliminate_chip"
                else:   #deadzone entire pile
                    for chip in self.rows[self.last_played_row]:
                        player = next((p for p in self.players if p.letter == chip), None)
                        if player:
                            player.dead_chips += 1
                    self.rows[self.last_played_row] = []
                    if self._current_player().count_total_chips() == 0:
                        self._check_player_elimination()
                    self.state = "choose_pile"
            else:
                self._determine_next_players()
            return True
        return False

    def _check_capture(self, row):
        played_pile = self.rows[row]
        return len(played_pile) > 1 and played_pile[-1] == played_pile[-2]

    def _determine_next_players(self):
        played_pile = self.rows[self.last_played_row]
        active_players = [p for p in self.players if not p.eliminated]
        active_player_letters = [p.letter for p in active_players]
        colors_in_pile = [color for color in set(played_pile) if not next(player.eliminated for player in self.players if player.letter == color)]

        if sorted(colors_in_pile) == sorted(active_player_letters):
            last_appearances = {color: len(played_pile) - 1 - played_pile[::-1].index(color)
                                for color in colors_in_pile}
            next_player_letter = min(last_appearances, key=last_appearances.get)
            self._set_next_player(next(p for p in active_players if p.letter == next_player_letter))
            self.state = "choose_pile"
        else:
            self.eligible_next_players = [p for p in active_players if p.letter not in colors_in_pile]
            if len(self.eligible_next_players) == 1:
                self._set_next_player(self.eligible_next_players[0])
                self.state = "choose_pile"
            else:
                self.state = "choose_next_player"

    def _choose_next_player(self, letter):
        if any(p.letter == letter for p in self.eligible_next_players):
            self._set_next_player(next(p for p in self.players if p.letter == letter))
            return True
        return False

    def _set_next_player(self, player):
        self.current_player_index = self.players.index(player)
        self.turn_history.append(player)
        self._check_player_elimination()

    def _check_player_elimination(self):
        while sum(self._current_player().chips.values()) == 0 and not self._current_player().eliminated:
          if self.turn_history and self._count_active_players()>1:
            self._current_player().eliminated = True
            #print(f"Player {self._current_player().letter} eliminated.")
            self.turn_history = [player for player in self.turn_history if player != self._current_player()]
            if len(self.turn_history)>0:
                self.current_player_index = self.players.index(self.turn_history[-1])
            else:
                self.current_player_index = next(i for i, player in enumerate(self.players) if not player.eliminated)
            #print(f"Current player is now: {self._current_player().letter}")
          else:
            env.done = True
            self.step(0)
            return True
        return False  # Indicate that the game is not over

    def _eliminate_chip(self, chip_letter):
        captured_pile = self.rows[self.last_played_row]
        #check if the chip being killed after a capture is actually in the pile
        if chip_letter in captured_pile:
            player = next((p for p in self.players if p.letter == chip_letter), None) #find the player whose chip we are trying to kill
            player.dead_chips += 1 #increment his killed chip count
            for chip in captured_pile: #add all chips other than the eliminated chip into current_player's pocket
                self._current_player().chips[chip] += 1
            self._current_player().chips[chip_letter] -= 1
            #empty the last played row and set state
            self.rows[self.last_played_row] = []
            self.state = "choose_pile"
            return True
        #the chip being eliminated is not in the pile, hence it is an illegal choice
        return False

    def _count_active_players(self):
        return sum(1 for player in self.players if not player.eliminated)

    def is_game_over(self):
        return self._count_active_players() <= 1

    def render(self):
        pass

    def close(self):
        pygame.quit()

    class Player:
        def __init__(self, letter, max_chips, player_letters):
            self.letter = letter
            self.chips = {l: max_chips if l == letter else 0 for l in player_letters}
            self.dead_chips = 0
            self.eliminated = False
            self.reward = 0

        def has_only_one_chip_type(self):
            return sum(1 for count in self.chips.values() if count > 0) == 1

        def get_only_chip_type(self):
            return next(letter for letter, count in self.chips.items() if count > 0)

        def count_total_chips(self):
            return sum(self.chips.values())