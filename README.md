# Modular Genetic Serpant

**Modular GS** is a genetic algorithm–based project that evolves a modular neural network to play the classic Snake game. The project employs a unique architecture where multiple subnetworks process distinct aspects of the game state (danger signals, food position, direction, and ray-casting sensor data) and combine their outputs to make decisions. In addition, parallel fitness evaluation and adaptive mutation help improve convergence.

---

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
  - Tournament selection (with improvements for diversity)
  - Uniform segmented crossover that mixes genetic material within each module
  - Adaptive mutation with decaying rate and strength

- **Parallel Processing:**  
  Uses Python’s multiprocessing to evaluate genomes concurrently for faster convergence.

- **Dynamic Simulation:**  
  The game adapts the simulation step limit based on performance, balancing exploration and survival.

---

## Project Structure

- **`main.py`**  
  Contains the `SnakeGame` class and the core game logic including state representation, as well as a version of an agent with training routines.  
  _This file holds the stable versions of our implementations._

- **`experimental.py`**  
  Contains the neural network modular architecture, genome decoding, genetic algorithm functions (including improved crossover and tournament selection), and the training loop.  
  _This file is used for experimenting with new ideas to develop a robust agent._

- **`new_approach.py`**  
  This thing is a new way of making an agent that is new to me, it makes a Deep Q-Netwotk that trys to predict the state of the game.
  _This is not the original idea of using a genetic algorithm but but based on this I will try convert it to something that the gen alg will be able to use._

- **`play_agent_dqn.py`**  
  Now this is the same as agent_in_action but for our new agent architecture.  


---

## Requirements

- Python 3.7 or higher
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [Bokeh](https://docs.bokeh.org/) (and [bokeh_sampledata](https://pypi.org/project/bokeh_sampledata/) if using sample data)
- [Holoviews](http://holoviews.org/) (optional, for visualization)