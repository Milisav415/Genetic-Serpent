import sys
import pygame
import numpy as np
from main import SnakeGame            # Ensure main.py is in the same directory and contains SnakeGame
from experimental import get_action, create_random_genome, GENOME_LENGTH

def load_best_genome():
    """
    Attempts to load a saved best genome from disk.
    If not found, returns a random genome.
    """
    try:
        best_genome = np.load("best_genome.npy")
        print("Loaded best genome from file.")
    except Exception as e:
        print("Could not load best genome; using a random genome instead.")
        best_genome = create_random_genome()
    return best_genome

def main():
    # Initialize Pygame and set up display constants.
    pygame.init()
    CELL_SIZE = 20
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Modular Genetic Serpant - Real Time Agent Play")
    clock = pygame.time.Clock()

    # Create a game instance.
    game = SnakeGame(w=GRID_WIDTH, h=GRID_HEIGHT)
    game.reset()

    # Load the best genome (or use a random one if not available).
    best_genome = load_best_genome()

    # Define some colors.
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    running = True
    while running:
        # Handle window events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Retrieve the current state (using the richer state representation).
        state = game.get_state_2()  # Use get_state_2() for the 15-element vector.
        action = get_action(state, best_genome)
        game.update(action)

        # Clear the screen.
        screen.fill(BLACK)

        # Draw the food.
        food_x, food_y = game.food
        pygame.draw.rect(screen, RED, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw the snake.
        for segment in game.snake:
            seg_x, seg_y = segment
            pygame.draw.rect(screen, GREEN, (seg_x * CELL_SIZE, seg_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Optionally, draw grid lines.
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(screen, WHITE, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, WHITE, (0, y), (SCREEN_WIDTH, y))

        # Update the display.
        pygame.display.flip()

        # Check for game over.
        if game.game_over():
            print("Game Over! Final Score:", game.score)
            running = False

        # Limit frame rate to control the speed of gameplay.
        clock.tick(10)  # 10 frames per second

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
