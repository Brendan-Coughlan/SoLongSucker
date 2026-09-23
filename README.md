# So Long Sucker (SLS)

This repository contains a reimplementation of **So Long Sucker (SLS)**, a turn-based strategy game developed by John Nash and others.

The project builds on a previous implementation of the game, with the goal of improving the codebase and reinforcement learning environment while keeping the game logic and benchmark behavior as close to the previous implementation as possible.

## Previous Implementation

The original implementation can be found here:

[MedantSharan/SoLongSucker](https://github.com/MedantSharan/SoLongSucker)

The original project provided a Pygame implementation of So Long Sucker and formed the basis for the environment used in this repository.

## About the Game

So Long Sucker is a turn-based strategy game in which players place chips in piles and attempt to capture other players' chips. The game continues until only one player remains.

The full game involves making and breaking informal coalitions and includes a chip-transfer system that makes the game significantly more complex. The chip-transfer system is not currently implemented.

## Features

- Supports four players, represented by the letters A, B, C, and D.
- Players place chips into piles during their turns.
- Players can capture piles by placing two consecutive chips of the same type.
- Players are eliminated when they can no longer continue playing.
- The game determines the next player based on the chips in the previously played pile.
- Gymnasium-based reinforcement learning environment.
- Random-agent baseline for evaluating the environment.
- Experiment logging and reward/episode-length plotting.

## Game States

1. **Choose Pile** — The current player chooses a pile in which to play.
2. **Choose Chip** — The player selects which type of chip to place.
3. **Choose Next Player** — When required, the current player selects the next eligible player.
4. **Eliminate Chip** — Following a capture, the player selects a chip type to eliminate.

## Current Work

This version refactors the previous implementation with a focus on:

- Migrating the environment to Gymnasium.
- Improving the project structure.
- Adding Python type annotations.
- Refactoring the environment and player logic.
- Using explicit game states.
- Reproducing the original random-agent benchmark.
- Adding experiment logging and visualization.
- Preparing the environment for reinforcement learning experiments.

The game logic and benchmark behavior are kept as close to the previous implementation as possible unless otherwise documented.

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

The current experiment runs multiple games using random agents and records their rewards and episode lengths for use as a baseline in later reinforcement learning experiments.