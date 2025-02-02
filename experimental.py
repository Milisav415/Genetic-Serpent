import numpy as np
import random
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from main import SnakeGame

# -------------------------------
# Neural Network / GA Parameters
# -------------------------------
# For the state vector from get_state_1, INPUT_SIZE remains 11.
INPUT_SIZE = 11
# Define two hidden layers.
HIDDEN_SIZE_1 = 16
HIDDEN_SIZE_2 = 16
OUTPUT_SIZE = 3

# Genome length calculation:
# Segment 1: Weights from Input to Hidden1: HIDDEN_SIZE_1 * INPUT_SIZE
# Segment 2: Biases for Hidden1: HIDDEN_SIZE_1
# Segment 3: Weights from Hidden1 to Hidden2: HIDDEN_SIZE_2 * HIDDEN_SIZE_1
# Segment 4: Biases for Hidden2: HIDDEN_SIZE_2
# Segment 5: Weights from Hidden2 to Output: OUTPUT_SIZE * HIDDEN_SIZE_2
# Segment 6: Biases for Output: OUTPUT_SIZE
GENOME_LENGTH = (HIDDEN_SIZE_1 * INPUT_SIZE + HIDDEN_SIZE_1 +
                 HIDDEN_SIZE_2 * HIDDEN_SIZE_1 + HIDDEN_SIZE_2 +
                 OUTPUT_SIZE * HIDDEN_SIZE_2 + OUTPUT_SIZE)

# -------------------------------
# Neural Network and Genome Setup
# -------------------------------
def decode_genome(genome):
    """
    Given a 1D genome, decode it into the parameters for the neural network
    with an extra hidden layer.
    """
    # Segment 1: Weights from Input to Hidden1.
    w1_end = HIDDEN_SIZE_1 * INPUT_SIZE
    # Segment 2: Biases for Hidden1.
    b1_end = w1_end + HIDDEN_SIZE_1
    # Segment 3: Weights from Hidden1 to Hidden2.
    w2_end = b1_end + HIDDEN_SIZE_2 * HIDDEN_SIZE_1
    # Segment 4: Biases for Hidden2.
    b2_end = w2_end + HIDDEN_SIZE_2
    # Segment 5: Weights from Hidden2 to Output.
    w3_end = b2_end + OUTPUT_SIZE * HIDDEN_SIZE_2
    # Segment 6: Biases for Output.
    b3_end = w3_end + OUTPUT_SIZE

    w1 = genome[0:w1_end].reshape((HIDDEN_SIZE_1, INPUT_SIZE))
    b1 = genome[w1_end:b1_end].reshape((HIDDEN_SIZE_1,))
    w2 = genome[b1_end:w2_end].reshape((HIDDEN_SIZE_2, HIDDEN_SIZE_1))
    b2 = genome[w2_end:b2_end].reshape((HIDDEN_SIZE_2,))
    w3 = genome[b2_end:w3_end].reshape((OUTPUT_SIZE, HIDDEN_SIZE_2))
    b3 = genome[w3_end:b3_end].reshape((OUTPUT_SIZE,))
    return w1, b1, w2, b2, w3, b3

def nn_predict(state, genome):
    """
    Forward pass through the neural network with an extra hidden layer.
    """
    w1, b1, w2, b2, w3, b3 = decode_genome(genome)
    layer1 = np.tanh(np.dot(w1, state) + b1)
    layer2 = np.tanh(np.dot(w2, layer1) + b2)
    output = np.dot(w3, layer2) + b3
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
        state = game.get_state_1()
        action = get_action(state, genome)
        game.update(action)
        steps += 1
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

def crossover_segments(parent1, parent2):
    """
    Perform a segmented single-point crossover between two parent genomes.
    Each segment corresponds to a different set of neural network parameters:
      - Segment 1: Weights from Input to Hidden1 (indices 0 to w1_end)
      - Segment 2: Biases for Hidden1 (indices w1_end to b1_end)
      - Segment 3: Weights from Hidden1 to Hidden2 (indices b1_end to w2_end)
      - Segment 4: Biases for Hidden2 (indices w2_end to b2_end)
      - Segment 5: Weights from Hidden2 to Output (indices b2_end to w3_end)
      - Segment 6: Biases for Output (indices w3_end to b3_end)
    """
    w1_end = HIDDEN_SIZE_1 * INPUT_SIZE
    b1_end = w1_end + HIDDEN_SIZE_1
    w2_end = b1_end + HIDDEN_SIZE_2 * HIDDEN_SIZE_1
    b2_end = w2_end + HIDDEN_SIZE_2
    w3_end = b2_end + OUTPUT_SIZE * HIDDEN_SIZE_2
    b3_end = w3_end + OUTPUT_SIZE

    seg1_length = w1_end
    cp1 = np.random.randint(0, seg1_length)
    child_seg1 = np.concatenate((parent1[0:cp1], parent2[cp1:w1_end]))

    seg2_length = HIDDEN_SIZE_1
    cp2 = np.random.randint(0, seg2_length)
    child_seg2 = np.concatenate((parent1[w1_end:w1_end+cp2], parent2[w1_end+cp2:b1_end]))

    seg3_length = HIDDEN_SIZE_2 * HIDDEN_SIZE_1
    cp3 = np.random.randint(0, seg3_length)
    child_seg3 = np.concatenate((parent1[b1_end:b1_end+cp3], parent2[b1_end+cp3:w2_end]))

    seg4_length = HIDDEN_SIZE_2
    cp4 = np.random.randint(0, seg4_length)
    child_seg4 = np.concatenate((parent1[w2_end:w2_end+cp4], parent2[w2_end+cp4:b2_end]))

    seg5_length = OUTPUT_SIZE * HIDDEN_SIZE_2
    cp5 = np.random.randint(0, seg5_length)
    child_seg5 = np.concatenate((parent1[b2_end:b2_end+cp5], parent2[b2_end+cp5:w3_end]))

    seg6_length = OUTPUT_SIZE
    cp6 = np.random.randint(0, seg6_length)
    child_seg6 = np.concatenate((parent1[w3_end:w3_end+cp6], parent2[w3_end+cp6:b3_end]))

    child = np.concatenate((child_seg1, child_seg2, child_seg3, child_seg4, child_seg5, child_seg6))
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
    generations = 300
    best_genome, best_fitness, fitness_history, avg_history = genetic_algorithm(generations=generations)
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
        state = game.get_state_1()
        action = get_action(state, best_genome)
        game.update(action)
        steps += 1
        print("Score:", game.score, "Head:", game.snake[0])
    print("Final Score:", game.score)

if __name__ == "__main__":
    main()
