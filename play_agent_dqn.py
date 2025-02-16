import pygame
import numpy as np
from keras.models import load_model
from keras.losses import mean_squared_error  # Add this import
from main import SnakeGame  # Ensure this matches your SnakeGame implementation

# --- Game Visualization Settings ---
GRID_SIZE = 20
CELL_SIZE = 25  # Size of each grid cell in pixels
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

COLORS = {
    'background': (30, 30, 30),
    'snake': (0, 255, 0),
    'head': (255, 165, 0),  # Orange head
    'food': (255, 0, 0),
    'text': (255, 255, 255)
}


class SnakeVisualizer:
    def __init__(self, model_path, state_type='get_state_2'):
        # Initialize game and AI components
        self.game = SnakeGame(w=GRID_SIZE, h=GRID_SIZE)
        # Load model with custom objects mapping
        self.model = load_model(
            model_path,
            custom_objects={'mse': mean_squared_error}
        )
        self.state_type = state_type

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("dqn agent in action")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def get_state(self):
        """Get current game state using the specified state representation"""
        if self.state_type == 'get_state_1':
            return np.array(self.game.get_state_1())
        return np.array(self.game.get_state_2())

    def draw_game(self):
        """Render game elements using Pygame"""
        self.screen.fill(COLORS['background'])

        # Draw snake
        for i, (x, y) in enumerate(self.game.snake):
            color = COLORS['head'] if i == 0 else COLORS['snake']
            pygame.draw.rect(self.screen, color,
                             (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1))

        # Draw food
        pygame.draw.rect(self.screen, COLORS['food'],
                         (self.game.food[0] * CELL_SIZE, self.game.food[1] * CELL_SIZE,
                          CELL_SIZE - 1, CELL_SIZE - 1))

        # Draw score
        score_text = self.font.render(f"Score: {self.game.score}", True, COLORS['text'])
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def run(self, max_steps=2000, fps=10):
        """Run the game visualization with AI control"""
        self.game.reset()
        running = True
        step = 0

        while running and step < max_steps:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get AI action
            state = self.get_state()
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            action_idx = np.argmax(q_values)
            action = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][action_idx]

            # Update game state
            self.game.update(action)
            self.draw_game()

            # Control game speed
            self.clock.tick(fps)

            # Check game over
            if self.game.game_over():
                print(f"Game Over! Final Score: {self.game.score}")
                running = False
            step += 1
        pygame.quit()


if __name__ == "__main__":
    # Initialize visualizer with your trained model
    visualizer = SnakeVisualizer(
        model_path="C:\\Users\\jm190\\PycharmProjects\\gen_alg\\dqn_snake_enhanced.h5",  # Path to your saved model
        state_type='get_state_2'  # Match your training state representation
    )

    # Run the visualization
    visualizer.run(fps=15)  # Adjust FPS for game speed