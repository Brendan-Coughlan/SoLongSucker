import pygame
import sys
import random

# Constants (unchanged)
MAX_CHIPS = 2
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
ROWS = 20
ROW_HEIGHT = (1/ROWS) * 550
PLAYER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
PLAYER_LETTERS = ['A', 'B', 'C', 'D']
BACKGROUND_COLOR = (125, 125, 125)

class Player:
    def __init__(self, letter, color):
        self.letter = letter
        self.color = color
        self.chips = {l: MAX_CHIPS if l == letter else 0 for l in PLAYER_LETTERS}
        self.dead_chips = 0
        self.eliminated = False

    def has_only_one_chip_type(self):
        return sum(1 for count in self.chips.values() if count > 0) == 1

    def get_only_chip_type(self):
        return next(letter for letter, count in self.chips.items() if count > 0)



class Game:
    def __init__(self):
        self.players = [Player(letter, color) for letter, color in zip(PLAYER_LETTERS, PLAYER_COLORS)]
        self.current_player = self.players[0]
        self.rows = [[] for _ in range(ROWS)]
        self.input_text = ""
        self.state = "choose_pile"  # States: "choose_pile", "choose_chip", "choose_next_player", "eliminate_chip"
        self.last_played_row = None
        self.eligible_next_players = []
        self.turn_history = [self.current_player]  # New: to keep track of who gave the turn to whom

    def play_chip(self, row, chip_letter):
        if 1 <= row <= ROWS and self.current_player.chips[chip_letter] > 0:
            self.rows[row-1].append(chip_letter)
            self.current_player.chips[chip_letter] -= 1
            self.last_played_row = row - 1
            if self.check_capture(row - 1):
                player = next((p for p in self.players if p.letter == chip_letter), None)
                if player and not player.eliminated: #do this only if capturing colored player is not eliminated otherwise entire pile is deadzoned
                    self.set_current_player(chip_letter) 
                self.state = "eliminate_chip"
            else:
                self.determine_next_players()
            return True
        return False

    def check_capture(self, row):
        played_pile = self.rows[row]
        return len(played_pile) > 1 and played_pile[-1] == played_pile[-2]

    def eliminate_chip(self, chip_letter):
        captured_pile = self.rows[self.last_played_row]
        if chip_letter in captured_pile:
            player = next((p for p in self.players if p.letter == chip_letter), None) #?
            if player and not player.eliminated:
                player.dead_chips += 1
                for chip in captured_pile:
                        self.current_player.chips[chip] += 1
                self.current_player.chips[chip_letter] -= 1
            else:
                # If the captured chip belongs to an eliminated player, move all chips to deadzone
                for chip in captured_pile:
                    player = next((p for p in self.players if p.letter == chip), None)
                    if player:
                        player.dead_chips += 1
            self.rows[self.last_played_row] = []
            self.state = "choose_pile"
            return True
        return False

    def determine_next_players(self): 
        played_pile = self.rows[self.last_played_row]
        active_players = [p for p in self.players if not p.eliminated]
        active_player_letters = [p.letter for p in active_players]
        colors_in_pile = [] #active colors in pile
        for color in set(played_pile):
            if not next(player.eliminated for player in self.players if player.letter == color):
                colors_in_pile.append(color)
        
        if sorted(colors_in_pile) == sorted(active_player_letters):  # All active colors represented
            last_appearances = {color: len(played_pile) - 1 - played_pile[::-1].index(color) 
                                for color in colors_in_pile if not next((p for p in self.players if p.letter == color), None).eliminated}
            next_player_letter = min(last_appearances, key=last_appearances.get)
            self.set_next_player(next(p for p in active_players if p.letter == next_player_letter))
            self.state = "choose_pile"
        else:  # Some active colors not represented
            active_players = [p for p in self.players if not p.eliminated]
            active_player_letters = [p.letter for p in active_players]
            self.eligible_next_players = [p for p in active_players if p.letter not in colors_in_pile]
            if len(self.eligible_next_players) == 1:  # Only one eligible player
                print("Only one eligible player " + self.eligible_next_players[0].letter)
                self.set_next_player(self.eligible_next_players[0])
                self.turn_history.append(self.eligible_next_players[0])
                self.state = "choose_pile"
            else:
                self.state = "choose_next_player"

    def choose_next_player(self, letter):
        if any(p.letter == letter for p in self.eligible_next_players):
            self.set_next_player(next(p for p in self.players if p.letter == letter))
            return True
        return False
    
    def set_current_player(self, letter):
        self.current_player = next(p for p in self.players if p.letter == letter)
    
    def set_next_player(self, player):
        # if self.turn_history[-1] != self.current_player:    
        #     self.turn_history.append(self.current_player)
        #     self.current_player = player
        #     #self.check_player_elimination()
        # else:
            self.current_player = player
            self.turn_history.append(self.current_player)
            self.check_player_elimination()
        

    def count_active_players(self):
        return sum(1 for player in self.players if not player.eliminated)
 
    def check_player_elimination(self):
        while sum(self.current_player.chips.values()) == 0 and not self.current_player.eliminated:
            if self.turn_history and self.count_active_players()>1 :
                print(self.current_player.letter + " got eliminated")
                self.current_player.eliminated = True
                self.turn_history = [player for player in self.turn_history if player != self.current_player] #remove all instances of eliminated player from turn history
                if len(self.turn_history) > 0:
                    self.current_player = self.turn_history[-1] #problem 1
                else:
                    self.current_player = next(player for player in self.players if not player.eliminated)
                print("Current player now is " + self.current_player.letter)
            else:
                return True  # Game over, only one player left and that is the winner
        return False

    def is_game_over(self):
        active_players = [p for p in self.players if not p.eliminated]
        return len(active_players) <= 1

def draw_game(screen, game):
    screen.fill(BACKGROUND_COLOR)

    # Draw rows
    font = pygame.font.Font(None, 20)
    for i, row in enumerate(game.rows):
        row_number = font.render(f"{i+1:2d}", True, (200, 200, 200))
        screen.blit(row_number, (10, i * ROW_HEIGHT + 10))
        
        for j, letter in enumerate(row):
            color = PLAYER_COLORS[PLAYER_LETTERS.index(letter)]
            player_eliminated =next(player.eliminated for player in game.players if player.letter == letter)
            if player_eliminated:
                color = (0,0,0)
            text = font.render(letter, True, color)
            screen.blit(text, (50 + j * 20, i * ROW_HEIGHT + 10))

    # Draw player info
    for i, player in enumerate(game.players):
        if not player.eliminated:
            font = pygame.font.Font(None, 24)
            text = font.render(f"Player {player.letter}:", True, player.color)
            screen.blit(text, (SCREEN_WIDTH - 200, 50 + i * 100))
            for j, (letter, count) in enumerate(player.chips.items()):
                if letter == player.letter:
                    chip_text = font.render(f"Own Chips: {count}", True, player.color)
                else:
                    chip_text = font.render(f"{letter}'s Chips: {count}", True, PLAYER_COLORS[PLAYER_LETTERS.index(letter)])
                screen.blit(chip_text, (SCREEN_WIDTH - 180, 70 + i * 100 + j * 20))
        else:
            font = pygame.font.Font(None, 24)
            text = font.render(f"Player {player.letter} is eliminated", True, (0,0,0))
            screen.blit(text, (SCREEN_WIDTH - 200, 50 + i * 100))

    # Draw current player indicator
    font = pygame.font.Font(None, 36)
    text = font.render(f"Current Player: {game.current_player.letter}", True, game.current_player.color)
    screen.blit(text, (50, SCREEN_HEIGHT - 100))

    # Draw input box with prompt
    font = pygame.font.Font(None, 28)
    pygame.draw.rect(screen, (200, 200, 200), (50, SCREEN_HEIGHT - 50, 700, 40))
    if game.input_text:
        text = font.render(game.input_text, True, (0, 0, 0))
    else:
        if game.state == "choose_pile":
            prompt = "Type which pile you want to play on and press Enter"
        elif game.state == "choose_chip":
            eligible_chips = [l for l in PLAYER_LETTERS if game.current_player.chips[l]!=0]
            prompt = f"Choose chip to play ({', '.join(eligible_chips)}) and press Enter"
        elif game.state == "choose_next_player":
            eligible_letters = [p.letter for p in game.eligible_next_players]
            prompt = f"Choose next player ({', '.join(eligible_letters)}) and press Enter"
        elif game.state == "eliminate_chip":
            captured_pile = game.rows[game.last_played_row]
            prompt = f"Choose chip to eliminate ({', '.join(set(captured_pile))}) and press Enter"
        text = font.render(prompt, True, (100, 100, 100))
    screen.blit(text, (55, SCREEN_HEIGHT - 45))

    # Draw deadzone
    font = pygame.font.Font(None, 24)
    deadzone_start_x = SCREEN_WIDTH - 200
    deadzone_start_y = SCREEN_HEIGHT - 150
    pygame.draw.rect(screen, (50, 50, 50), (deadzone_start_x, deadzone_start_y, 180, 130))
    
    text = font.render("Deadzone", True, (255, 255, 255))
    screen.blit(text, (deadzone_start_x + 50, deadzone_start_y + 10))

    for i, player in enumerate(game.players):
        text = font.render(f"{player.letter}: {player.dead_chips}", True, (0, 0, 0))
        pygame.draw.rect(screen, player.color, (deadzone_start_x + 10 + (i % 2) * 90, 
                                                deadzone_start_y + 40 + (i // 2) * 40, 80, 30))
        screen.blit(text, (deadzone_start_x + 15 + (i % 2) * 90, 
                           deadzone_start_y + 45 + (i // 2) * 40))

def main():
    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Chip Game")
    clock = pygame.time.Clock()

    print("Creating game instance...")
    game = Game()

    # Add a small delay to allow time for initialization
    pygame.time.wait(1000)

    print("Entering main game loop...")
    for player in game.turn_history:
        print(player.letter, " -> ", end='')
    print('\n')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or game.is_game_over():
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if game.state == "choose_pile":
                        if game.input_text.isdigit():
                            game.last_played_row = int(game.input_text)
                            if game.current_player.has_only_one_chip_type():
                                game.play_chip(game.last_played_row, game.current_player.get_only_chip_type())
                            else:
                                game.state = "choose_chip"
                    elif game.state == "choose_chip":
                        if game.input_text.upper() in PLAYER_LETTERS:
                            if game.play_chip(game.last_played_row, game.input_text.upper()):
                                if game.state == "choose_pile":  # Next player was automatically determined
                                    game.input_text = ""
                    elif game.state == "choose_next_player":
                        if game.input_text.upper() in PLAYER_LETTERS:
                            if game.choose_next_player(game.input_text.upper()):
                                for player in game.turn_history:
                                    print(player.letter, " -> ", end='')
                                print('\n')
                                game.state = "choose_pile"
                                # if game.check_player_elimination():
                                #     running = False
                                # else:
                                #     game.state="choose_pile"
                    elif game.state == "eliminate_chip":
                        if game.input_text.upper() in PLAYER_LETTERS:
                            if game.eliminate_chip(game.input_text.upper()):
                                game.input_text = ""
                                # if game.check_player_elimination():
                                #     running = False
                                # else:
                                #     game.state="choose_pile"
                    game.input_text = ""
                elif event.key == pygame.K_BACKSPACE:
                    game.input_text = game.input_text[:-1]
                elif game.state == "choose_pile" and event.unicode.isdigit():
                    game.input_text += event.unicode
                elif (game.state in ["choose_chip", "choose_next_player", "eliminate_chip"] and 
                      event.unicode.upper() in PLAYER_LETTERS):
                    game.input_text = event.unicode.upper()

        screen.fill(BACKGROUND_COLOR)
        draw_game(screen, game)
        pygame.display.flip()
        clock.tick(60)

        if game.is_game_over():
            font = pygame.font.Font(None, 48)
            winner = next((p for p in game.players if not p.eliminated), None)
            text = font.render(f"Game Over! Player {winner.letter} wins!", True, (255, 255, 255))
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2))
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False

    print("Game loop ended. Quitting...")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

#problem 1: A and B play game among themselves and both get eliminated which means turn history becomes empty => random player allocated