import numpy as np
import random
import math
import matplotlib.pyplot as plt

# -------------------------------
# Snake Game Environment
# -------------------------------
class SnakeGame:
    def __init__(self, w=20, h=20):
        self.width = w
        self.height = h
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # initial direction: right
        self.place_food()
        self.score = 0
        self.frame_iteration = 0

    def place_food(self):
        available_positions = [(x, y) for x in range(self.width) for y in range(self.height)
                               if (x, y) not in self.snake]
        self.food = random.choice(available_positions)

    def is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        x, y = point
        # Check wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Check self collision
        if point in self.snake[1:]:
            return True
        return False

    def update(self, action):
        """
        Update the game state given an action.
        The action is a one-hot vector representing:
            [1, 0, 0] -> keep going straight
            [0, 1, 0] -> turn right
            [0, 0, 1] -> turn left
        """
        # Define directions in order: Up, Right, Down, Left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        current_index = directions.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # straight: no change
            new_index = current_index
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_index = (current_index + 1) % 4
        elif np.array_equal(action, [0, 0, 1]):  # left turn
            new_index = (current_index - 1) % 4
        else:
            new_index = current_index

        self.direction = directions[new_index]
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.snake.insert(0, new_head)

        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        self.frame_iteration += 1

    def normalized_distance(self, direction):
        """
        Compute the normalized distance from the snake's head in a given direction
        until a collision occurs. The value is steps_taken / max_possible_steps.
        For cardinal directions the maximum steps (ignoring self-collision) are:
            - Right: width - head_x - 1
            - Left: head_x
            - Down: height - head_y - 1
            - Up: head_y
        """
        head_x, head_y = self.snake[0]
        steps = 0
        current_point = (head_x, head_y)
        while True:
            next_point = (current_point[0] + direction[0], current_point[1] + direction[1])
            if self.is_collision(next_point):
                break
            steps += 1
            current_point = next_point

        # Compute maximum possible steps in that direction (only considering walls)
        if direction == (1, 0):  # right
            max_steps = self.width - head_x - 1
        elif direction == (-1, 0):  # left
            max_steps = head_x
        elif direction == (0, 1):  # down
            max_steps = self.height - head_y - 1
        elif direction == (0, -1):  # up
            max_steps = head_y
        else:
            max_steps = steps if steps != 0 else 1  # fallback

        if max_steps == 0:
            return 0.0
        return steps / max_steps

    def get_state_2(self):
        """
        Returns a 7-element state vector encoding:
        1. Danger signals: normalized distance to collision in 3 directions (straight, right, left).
        2. Direction: the snake's current movement encoded as sin(angle) and cos(angle).
        3. Food position: the relative position (dx, dy) of the food normalized by the board dimensions.
        """
        # 1. Danger signals.
        # Define directions relative to the current heading.
        # "Straight" is the current direction.
        # "Right" and "Left" are 90° rotations of the current direction.
        dx, dy = self.direction
        straight = self.direction
        right = (dy, -dx)  # 90° clockwise rotation.
        left = (-dy, dx)  # 90° counterclockwise rotation.

        danger_straight = self.normalized_distance(straight)
        danger_right = self.normalized_distance(right)
        danger_left = self.normalized_distance(left)

        # 2. Direction encoded as sin and cos.
        angle = math.atan2(dy, dx)
        dir_sin = math.sin(angle)
        dir_cos = math.cos(angle)

        # 3. Relative food position (continuous coordinates).
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        food_dx = (food_x - head_x) / self.width
        food_dy = (food_y - head_y) / self.height

        state = np.array([danger_straight, danger_right, danger_left,
                          dir_sin, dir_cos,
                          food_dx, food_dy], dtype=float)
        return state

    def get_state_1(self):
        """
        Returns an 11-element state vector:
          - Danger straight, right, left (1 if danger, else 0)
          - Current direction (one-hot: up, right, down, left)
          - Food location relative to head (food is up, down, left, right)
        """
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dir_index = directions.index(self.direction)

        def danger_in_direction(direction):
            head_x, head_y = self.snake[0]
            next_point = (head_x + direction[0], head_y + direction[1])
            return 1 if self.is_collision(next_point) else 0

        danger_straight = danger_in_direction(self.direction)
        danger_right = danger_in_direction(directions[(dir_index + 1) % 4])
        danger_left = danger_in_direction(directions[(dir_index - 1) % 4])

        dir_up = 1 if self.direction == (0, -1) else 0
        dir_right = 1 if self.direction == (1, 0) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            food_up,
            food_down,
            food_left,
            food_right
        ]
        return np.array(state, dtype=int)

    def game_over(self):
        return self.is_collision()

# -------------------------------
# Neural Network and Genome Setup
# -------------------------------
# Define network sizes and genome length.
INPUT_SIZE = 7
HIDDEN_SIZE = 32
OUTPUT_SIZE = 3

# Genome: weights and biases flattened.
GENOME_LENGTH = HIDDEN_SIZE * INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE * HIDDEN_SIZE + OUTPUT_SIZE

def decode_genome(genome):
    """
    Given a 1D genome, decode into network parameters.
    """
    w1_end = HIDDEN_SIZE * INPUT_SIZE
    b1_end = w1_end + HIDDEN_SIZE
    w2_end = b1_end + OUTPUT_SIZE * HIDDEN_SIZE
    b2_end = w2_end + OUTPUT_SIZE

    w1 = genome[0:w1_end].reshape((HIDDEN_SIZE, INPUT_SIZE))
    b1 = genome[w1_end:b1_end].reshape((HIDDEN_SIZE,))
    w2 = genome[b1_end:w2_end].reshape((OUTPUT_SIZE, HIDDEN_SIZE))
    b2 = genome[w2_end:b2_end].reshape((OUTPUT_SIZE,))
    return w1, b1, w2, b2

def nn_predict(state, genome):
    """
    Forward pass through the neural network.
    """
    w1, b1, w2, b2 = decode_genome(genome)
    layer1 = np.tanh(np.dot(w1, state) + b1)
    output = np.dot(w2, layer1) + b2
    return output

def get_action(state, genome):
    """
    Returns a one-hot vector for the action selected by the network.
    Action encoding:
        0 -> straight, 1 -> right turn, 2 -> left turn
    """
    prediction = nn_predict(state, genome)
    action_index = np.argmax(prediction)
    action = [0, 0, 0]
    action[action_index] = 1
    return np.array(action)

# -------------------------------
# Genetic Algorithm Components
# -------------------------------
def evaluate_genome(genome, max_steps=200):
    """
    Evaluate a genome by running a game simulation.
    The fitness is a combination of the game score and the number of frames survived.
    """
    game = SnakeGame()
    game.reset()
    steps = 0
    while not game.game_over() and steps < max_steps:
        state = game.get_state_2()
        action = get_action(state, genome)
        game.update(action)
        steps += 1
    # Fitness: reward food eaten heavily and also give credit for survival time.
    fitness = game.score * 100 + game.frame_iteration
    return fitness

def create_random_genome():
    """
    Returns a random genome vector.
    """
    return np.random.uniform(-1, 1, GENOME_LENGTH)

def tournament_selection(population, fitnesses, tournament_size=5):
    """
    Select one genome from the population using tournament selection.
    """
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    best = None
    best_fitness = -np.inf
    for i in selected_indices:
        if fitnesses[i] > best_fitness:
            best = population[i]
            best_fitness = fitnesses[i]
    return best

def crossover(parent1, parent2):
    """
    Perform a single-point crossover between two parent genomes.
    """
    child = np.copy(parent1)
    crossover_point = np.random.randint(0, GENOME_LENGTH)
    child[crossover_point:] = parent2[crossover_point:]
    return child

def crossover_segments(parent1, parent2):
    """
    Perform a segmented single-point crossover between two parent genomes.
    Each segment corresponds to a different set of neural network parameters:
      - Segment 1: Weights from Input to Hidden (indices 0 to w1_end)
      - Segment 2: Biases for the Hidden layer (indices w1_end to b1_end)
      - Segment 3: Weights from Hidden to Output (indices b1_end to w2_end)
      - Segment 4: Biases for the Output layer (indices w2_end to b2_end)
    """
    # Define segment boundaries based on the network architecture.
    w1_end = HIDDEN_SIZE * INPUT_SIZE           # Input -> Hidden weights (16 * 11 = 176)
    b1_end = w1_end + HIDDEN_SIZE                # Hidden biases (176 + 16 = 192)
    w2_end = b1_end + OUTPUT_SIZE * HIDDEN_SIZE   # Hidden -> Output weights (192 + 48 = 240)
    b2_end = w2_end + OUTPUT_SIZE                # Output biases (240 + 3 = 243)

    # Segment 1: Weights from Input to Hidden
    seg1_length = w1_end  # 176
    cp1 = np.random.randint(0, seg1_length)
    child_seg1 = np.concatenate((parent1[0:cp1], parent2[cp1:w1_end]))

    # Segment 2: Biases for the Hidden layer
    seg2_length = HIDDEN_SIZE  # 16
    cp2 = np.random.randint(0, seg2_length)
    child_seg2 = np.concatenate((parent1[w1_end:w1_end+cp2], parent2[w1_end+cp2:b1_end]))

    # Segment 3: Weights from Hidden to Output
    seg3_length = OUTPUT_SIZE * HIDDEN_SIZE  # 48
    cp3 = np.random.randint(0, seg3_length)
    child_seg3 = np.concatenate((parent1[b1_end:b1_end+cp3], parent2[b1_end+cp3:w2_end]))

    # Segment 4: Biases for the Output layer
    seg4_length = OUTPUT_SIZE  # 3
    cp4 = np.random.randint(0, seg4_length)
    child_seg4 = np.concatenate((parent1[w2_end:w2_end+cp4], parent2[w2_end+cp4:b2_end]))

    # Concatenate all segments to form the full child genome.
    child = np.concatenate((child_seg1, child_seg2, child_seg3, child_seg4))
    return child

def mutate(genome, mutation_rate=0.01, mutation_strength=0.1):
    """
    Mutate a genome by adding Gaussian noise to some of its genes.
    """
    for i in range(len(genome)):
        if np.random.rand() < mutation_rate:
            genome[i] += np.random.normal(0, mutation_strength)
    return genome

def genetic_algorithm(population_size=150, generations=100):
    """
    Main loop of the genetic algorithm.
    Returns the best genome, its fitness, and a history of the best fitness per generation.
    """
    # Create initial population.
    population = [create_random_genome() for _ in range(population_size)]
    best_genome = None
    best_fitness = -np.inf
    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(generations):
        fitnesses = [evaluate_genome(genome) for genome in population]
        gen_best = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(gen_best)
        avg_fitness_history.append(avg_fitness)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[np.argmax(fitnesses)]
        print(f"Generation {gen} Best Fitness: {gen_best}, Average Fitness: {avg_fitness}")

        new_population = []
        # Elitism: preserve the best genome.
        elite = best_genome.copy()
        new_population.append(elite)
        # Create new individuals until the population is refilled.
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover_segments(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    return best_genome, best_fitness, best_fitness_history, avg_fitness_history

def plot_avg_fitness_history(avg_fitness_history):
    """
    Plots the average fitness over generations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(avg_fitness_history)), avg_fitness_history, marker='o', linestyle='-', color='green')
    plt.title("Average Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness Score")
    plt.grid(True)
    plt.show()

# -------------------------------
# Main Function
# -------------------------------
def main():
    generations = 100  # number of generations for the GA
    best_genome, best_fitness, fitness_history, avg_history = genetic_algorithm(generations=generations)
    print("Training complete.")
    print("Best Fitness:", best_fitness)

    # Plot the average fitness over generations
    plot_avg_fitness_history(avg_history)

    # Plot the best fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), fitness_history, marker='o', linestyle='-', color='b')
    plt.title("Best Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.show()

    # Run a final simulation using the best genome.
    game = SnakeGame()
    game.reset()
    steps = 0
    while not game.game_over() and steps < 500:
        state = game.get_state_1()
        action = get_action(state, best_genome)
        game.update(action)
        steps += 1
        # For demonstration purposes, print the current score and snake head position.
        print("Score:", game.score, "Head:", game.snake[0])
    print("Final Score:", game.score)

if __name__ == "__main__":
    main()
