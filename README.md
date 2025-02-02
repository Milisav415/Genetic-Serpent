# Modular Genetic Serpant

Modular GS is a genetic algorithm–based project that evolves a modular neural network to play the classic Snake game. The project employs a unique architecture where multiple subnetworks process distinct aspects of the game state (danger signals, food position, direction, and ray-casting sensor data) and combine their outputs to make decisions. In addition, parallel fitness evaluation and adaptive mutation help improve convergence.

## Features

- **Modular Neural Network:**
  The network is split into several subnetworks:
  - **Danger Subnetwork:** Processes three danger signals.
  - **Food Subnetwork:** Processes food position (dx, dy).
  - **Direction Subnetwork:** Processes direction information (sin, cos).
  - **Ray Subnetwork:** Processes eight ray-casting sensor values.
  
  These outputs are concatenated and fed into a combined hidden layer before producing an output that determines the snake's action.

- **Genetic Algorithm:**
  Evolve the neural network weights using:
  - Tournament selection (with improvements for diversity).
  - Uniform segmented crossover that mixes genetic material within each module.
  - Adaptive mutation with decaying rate and strength.

- **Parallel Processing:**  
  Uses Python’s multiprocessing to evaluate genomes concurrently for faster convergence.

- **Dynamic Simulation:**  
  The game adapts the simulation step limit based on performance, balancing exploration and survival.

## Project Structure

- `main.py`  
  Contains the `SnakeGame` class and the core game logic including state representation, and a version of an agent and trainig it.
  Here we put the stable wersions of what we bould in experimental.py

- `experimental.py`  
  Contains the neural network modular architecture, genome decoding, genetic algorithm functions (including improved crossover and tournament selection), and the training loop.
  As the name implies here we do all sort of ideas to se if we can come up with a good agent.

## Requirements

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Multiprocessing](https://multiprocessing.org/)
- [Bokeh](https://docs.bokeh.org/) (and [bokeh_sampledata](https://pypi.org/project/bokeh_sampledata/) if using sample data)
- [Holoviews](http://holoviews.org/) (optional, if using for visualization)

## Getting Started

**1. Clone the Repository:**
git clone https://github.com/yourusername/modular-evosnake.git
cd modular-Genetic-Serpant

**2. Run the Training:**
python experimental.py

**3. Watch the Agent Play:**
After training, the best genome is used to simulate a Snake game. The final game score and snake behavior will be printed to the console.

## Code Overview

**SnakeGame Class:**
Implements the Snake game including snake movement, collision detection, food placement, and state representation. Two state representations are available:
-get_state_1(): An 11-element state vector with binary danger signals, one-hot direction, and food location.
-get_state_2(): A 15-element state vector that includes richer features such as normalized distances and ray-casting sensor data.

## Modular Neural Network

**Subnetworks:**
The network splits the state into four parts (danger, food, direction, rays) and processes each with its own weights and biases.

**Combined Hidden Layer:**
Outputs from each subnetwork are concatenated and fed into a combined hidden layer, which then connects to the output layer that determines the snake’s next action.

## Genetic Algorithm

**Fitness Evaluation:**
Simulates the Snake game and calculates fitness based on game score and survival duration.

**Tournament Selection:**
Improved with stochastic elements to balance exploitation and exploration.

**Uniform Segmented Crossover:**
Genes are mixed uniformly within each module boundary for a finer blending of traits.

**Adaptive Mutation:**
Mutation rate and strength decay over generations for a balance between exploration and stability.

**Parallel Processing:**
Uses the multiprocessing module to evaluate many genomes concurrently.
