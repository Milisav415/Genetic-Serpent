import numpy as np
import random
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from main import SnakeGame

# -------------------------------
# Neural Network Parameters
# -------------------------------
# Modular subnetwork architecture
STATE_SIZE = 15  # Using get_state_2()
SUB_NET_SIZES = {
    'danger': (3, 6),    # 3 inputs (danger distances) → 8 neurons
    'food': (2, 6),      # 2 inputs (food dx/dy) → 8 neurons
    'direction': (2, 6), # 2 inputs (sin/cos) → 8 neurons
    'rays': (8, 6)       # 8 ray distances → 8 neurons
}
HIDDEN_SIZE = 24         # Combined hidden layer
OUTPUT_SIZE = 3


# Calculate genome length
GENOME_LENGTH = (
    # Danger subnetwork
    SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
    # Food subnetwork
    SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
    # Direction subnetwork
    SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1] +
    # Ray subnetwork
    SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1] + SUB_NET_SIZES['rays'][1] +
    # Combined layers
    (sum([v[1] for v in SUB_NET_SIZES.values()])) * HIDDEN_SIZE + HIDDEN_SIZE +
    HIDDEN_SIZE * OUTPUT_SIZE + OUTPUT_SIZE
)

# -------------------------------
# Neural Network Functions
# -------------------------------
def decode_genome(genome):
    """Decode genome into modular subnetwork weights/biases"""
    ptr = 0
    params = {}

    # Danger subnetwork (3 → 8)
    w1_danger_size = SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1]
    params['w1_danger'] = genome[ptr:ptr + w1_danger_size].reshape(SUB_NET_SIZES['danger'][::-1])
    ptr += w1_danger_size
    params['b1_danger'] = genome[ptr:ptr + SUB_NET_SIZES['danger'][1]]
    ptr += SUB_NET_SIZES['danger'][1]

    # Food subnetwork (2 → 8)
    w1_food_size = SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1]
    params['w1_food'] = genome[ptr:ptr + w1_food_size].reshape(SUB_NET_SIZES['food'][::-1])
    ptr += w1_food_size
    params['b1_food'] = genome[ptr:ptr + SUB_NET_SIZES['food'][1]]
    ptr += SUB_NET_SIZES['food'][1]

    # Direction subnetwork (2 → 8)
    w1_dir_size = SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1]
    params['w1_dir'] = genome[ptr:ptr + w1_dir_size].reshape(SUB_NET_SIZES['direction'][::-1])
    ptr += w1_dir_size
    params['b1_dir'] = genome[ptr:ptr + SUB_NET_SIZES['direction'][1]]
    ptr += SUB_NET_SIZES['direction'][1]

    # Ray subnetwork (8 → 8)
    w1_ray_size = SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1]
    params['w1_ray'] = genome[ptr:ptr + w1_ray_size].reshape(SUB_NET_SIZES['rays'][::-1])
    ptr += w1_ray_size
    params['b1_ray'] = genome[ptr:ptr + SUB_NET_SIZES['rays'][1]]
    ptr += SUB_NET_SIZES['rays'][1]

    # Combined layer (32 → 32)
    combined_size = sum([v[1] for v in SUB_NET_SIZES.values()])  # 8+8+8+8=32
    w2_size = combined_size * HIDDEN_SIZE
    params['w2'] = genome[ptr:ptr + w2_size].reshape((HIDDEN_SIZE, combined_size))
    ptr += w2_size
    params['b2'] = genome[ptr:ptr + HIDDEN_SIZE]
    ptr += HIDDEN_SIZE

    # Output layer (32 → 3)
    w3_size = HIDDEN_SIZE * OUTPUT_SIZE
    params['w3'] = genome[ptr:ptr + w3_size].reshape((OUTPUT_SIZE, HIDDEN_SIZE))
    ptr += w3_size
    params['b3'] = genome[ptr:ptr + OUTPUT_SIZE]

    return params


def nn_predict(state, genome):
    """Forward pass with modular subnetworks"""
    params = decode_genome(genome)

    # Split state into components
    danger = state[0:3]
    food = state[3:5]
    direction = state[5:7]
    rays = state[7:15]

    # Process subnetworks
    danger_out = np.tanh(np.dot(params['w1_danger'], danger) + params['b1_danger'])
    food_out = np.tanh(np.dot(params['w1_food'], food) + params['b1_food'])
    dir_out = np.tanh(np.dot(params['w1_dir'], direction) + params['b1_dir'])
    ray_out = np.tanh(np.dot(params['w1_ray'], rays) + params['b1_ray'])

    # Combine and process
    combined = np.concatenate([danger_out, food_out, dir_out, ray_out])
    hidden = np.tanh(np.dot(params['w2'], combined) + params['b2'])
    output = np.dot(params['w3'], hidden) + params['b3']

    return output

def get_action(state, genome):
    prediction = nn_predict(state, genome)
    return np.eye(OUTPUT_SIZE)[np.argmax(prediction)]

# -------------------------------
# Genetic Algorithm Components
# -------------------------------
def evaluate_genome(genome):
    """
    Evaluate a genome by running a game simulation.
    The fitness is a combination of the game score and the number of frames survived.
    """
    game = SnakeGame()
    game.reset()
    steps = 0
    max_steps = 300 + game.score * 50  # Dynamic step limit
    prev_food_distance = None

    while not game.game_over() and steps < max_steps:
        state = game.get_state_2()
        action = get_action(state, genome)
        game.update(action)
        steps += 1

        # Calculate food distance penalty/reward
        head_x, head_y = game.snake[0]
        food_distance = abs(head_x - game.food[0]) + abs(head_y - game.food[1])
        if prev_food_distance is not None:
            if food_distance < prev_food_distance:
                game.score += 0.1  # Reward moving toward food
        prev_food_distance = food_distance

    # Fitness = score + survival_time + food_efficiency
    fitness = (
        game.score * 100 +
        steps * 0.1 +
        (game.score / max(1, steps))  # Penalize slow scoring
    )
    return fitness

def create_random_genome():
    """
    Returns a random genome vector.
    """
    return np.random.uniform(-1, 1, GENOME_LENGTH)


def tournament_selection(population, fitnesses, tournament_size=10, best_prob=0.8):
    """
    Improved tournament selection that selects the best individual with a probability `best_prob`,
    otherwise selects a random individual from the tournament.
    """
    # Randomly select indices for the tournament.
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    # Pair each selected individual with its fitness.
    tournament = [(population[i], fitnesses[i]) for i in selected_indices]
    # Sort by fitness (assuming higher fitness is better).
    tournament.sort(key=lambda x: x[1], reverse=True)

    # With probability best_prob, select the best individual.
    if np.random.rand() < best_prob:
        return tournament[0][0]
    else:
        # Otherwise, randomly select one from the rest.
        return random.choice([ind for ind, _ in tournament[1:]])


def crossover_segments(parent1, parent2, uniform_rate=0.5):
    """
    Improved segmented crossover for the modular genome structure using uniform crossover.
    Each gene within a segment is chosen independently with probability 'uniform_rate' from parent1,
    otherwise from parent2.

    The segment boundaries are computed based on the architecture:
      - Danger subnetwork (w1_danger + b1_danger)
      - Food subnetwork (w1_food + b1_food)
      - Direction subnetwork (w1_dir + b1_dir)
      - Ray subnetwork (w1_ray + b1_ray)
      - Combined layer (w2 + b2)
      - Output layer (w3 + b3)
    """
    # Precompute segment boundaries (same as before)
    boundaries = [
        # Danger subnetwork (w1_danger + b1_danger)
        0,
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1],

        # Food subnetwork (w1_food + b1_food)
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1],
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1],

        # Direction subnetwork (w1_dir + b1_dir)
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1],
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1],

        # Ray subnetwork (w1_ray + b1_ray)
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1],
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1] +
        SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1] + SUB_NET_SIZES['rays'][1],

        # Combined layer (w2 + b2)
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1] +
        SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1] + SUB_NET_SIZES['rays'][1],
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1] +
        SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1] + SUB_NET_SIZES['rays'][1] +
        (sum([v[1] for v in SUB_NET_SIZES.values()])) * HIDDEN_SIZE + HIDDEN_SIZE,

        # Output layer (w3 + b3)
        SUB_NET_SIZES['danger'][0] * SUB_NET_SIZES['danger'][1] + SUB_NET_SIZES['danger'][1] +
        SUB_NET_SIZES['food'][0] * SUB_NET_SIZES['food'][1] + SUB_NET_SIZES['food'][1] +
        SUB_NET_SIZES['direction'][0] * SUB_NET_SIZES['direction'][1] + SUB_NET_SIZES['direction'][1] +
        SUB_NET_SIZES['rays'][0] * SUB_NET_SIZES['rays'][1] + SUB_NET_SIZES['rays'][1] +
        (sum([v[1] for v in SUB_NET_SIZES.values()])) * HIDDEN_SIZE + HIDDEN_SIZE,
        GENOME_LENGTH
    ]

    child = np.empty_like(parent1)
    # Loop over segments defined by boundaries
    for i in range(0, len(boundaries), 2):
        start = boundaries[i]
        end = boundaries[i + 1]
        # For each gene in the segment, choose gene from parent1 with probability uniform_rate,
        # else from parent2.
        for j in range(start, end):
            if np.random.rand() < uniform_rate:
                child[j] = parent1[j]
            else:
                child[j] = parent2[j]

    return child


def mutate(genome, generation, max_generations=100,
           initial_mutation_rate=0.1, final_mutation_rate=0.01,
           initial_strength=0.5, final_strength=0.05):
    """
    Adaptive mutation: Rate and strength decay exponentially over generations.
    """
    # Exponential decay
    decay_factor = (final_mutation_rate / initial_mutation_rate) ** (1 / max_generations)
    mutation_rate = initial_mutation_rate * (decay_factor ** generation)

    strength_decay = (final_strength / initial_strength) ** (1 / max_generations)
    mutation_strength = initial_strength * (strength_decay ** generation)

    for i in range(len(genome)):
        if np.random.rand() < mutation_rate:
            genome[i] += np.random.normal(0, mutation_strength)
    return genome

def genetic_algorithm(population_size=100, generations=100):
    """
    Main loop of the genetic algorithm.
    Returns the best genome, its fitness, and histories of best and average fitness per generation.
    """
    population = [create_random_genome() for _ in range(population_size)]
    best_genome = None
    best_fitness = -np.inf
    best_fitness_history = []
    avg_fitness_history = []

    # Create a process pool (use 0.75 of available CPUs)
    num_workers = max(1, int(mp.cpu_count() * 0.75))
    pool = mp.Pool(num_workers)

    for gen in range(generations):
        # Parallel fitness evaluation
        fitnesses = list(pool.map(evaluate_genome, population))
        gen_best = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness_history.append(gen_best)
        avg_fitness_history.append(avg_fitness)
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome = population[np.argmax(fitnesses)]
        print(f"Generation {gen} Best Fitness: {gen_best}, Average Fitness: {avg_fitness}")

        new_population = []
        elite = best_genome.copy()
        new_population.append(elite)
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover_segments(parent1, parent2)
            child = mutate(child, gen, generations)
            new_population.append(child)
        population = new_population

    pool.close()
    pool.join()
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
    generations = 1000
    population_size = 100
    best_genome, best_fitness, fitness_history, avg_history = genetic_algorithm(population_size=population_size, generations=generations)
    print("Training complete.")
    print("Best Fitness:", best_fitness)

    plot_avg_fitness_history(avg_history)

    plt.figure(figsize=(10, 6))
    plt.plot(range(generations), fitness_history, marker='o', linestyle='-', color='b')
    plt.title("Best Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Score")
    plt.grid(True)
    plt.show()

    game = SnakeGame()
    game.reset()
    steps = 0
    while not game.game_over() and steps < 500:
        state = game.get_state_2()
        action = get_action(state, best_genome)
        game.update(action)
        steps += 1
        print("Score:", game.score, "Head:", game.snake[0])
    print("Final Score:", game.score)

if __name__ == "__main__":
    main()
