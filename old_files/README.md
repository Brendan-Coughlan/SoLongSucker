# So Long Sucker (SLS) - A Strategy Game Made by John Nash

This is a Pygame implementation of a turn-based strategy game So Long Sucker (developed by John Nash among others) where players place chips in rows and aim to capture other players' chips. The game continues until only one player remains.
However, currently this is not a complete implementation of the game. The game involves making and breaking informal coalitions and has a chip transfer system which makes the game semi-sequential and way more complex. This chip transfer system has not been implemented yet.
At the time of committing this project, this is the first publicly available code for the game (even implemented partially) which can be played locally with four players.

## Features

- Supports up to 4 players, each with unique colors and letters.
- Players can capture other players' chips by placing two consecutive chips of the same type.
- Eliminated players' chips are moved to a "deadzone."
- The game automatically determines the next player based on the last played chips.
- Detailed UI displaying player information, current player, and game state.

## Gameplay

1. **Choose Pile**: Players choose a row to place their chip.
2. **Choose Chip**: Players select the type of chip to place if they have more than one type.
3. **Capture Chips**: If two consecutive chips of the same type are placed, the previous chip is captured.
4. **Eliminate Player**: If a player runs out of chips, they are eliminated.

## How to Run

1. Ensure you have Python installed on your system. This game requires Pygame, so you need to install it using pip:
   ```bash
   pip install pygame
2. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/MedantSharan/SoLongSucker.git
3. Run the game:
   ```bash
   python SLS.py
