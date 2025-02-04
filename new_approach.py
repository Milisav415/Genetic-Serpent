import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from main import SnakeGame


# --- Enhanced Environment Wrapper ---
class SnakeEnv:
    def __init__(self, grid_size=20, state_type='get_state_2'):  # Default to state_2
        self.game = SnakeGame(w=grid_size, h=grid_size)
        self.action_space = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.state_type = state_type
        self.state_size = 15 if state_type == 'get_state_2' else 11
        self.prev_score = 0
        self.prev_food_distance = None

    def reset(self):
        self.game.reset()
        self.prev_score = 0
        self.prev_food_distance = self._food_distance()
        return self._current_state()

    def step(self, action):
        self.game.update(action)
        done = self.game.game_over()
        reward = self._calculate_reward(done)
        return self._current_state(), reward, done

    def _current_state(self):
        if self.state_type == 'get_state_1':
            return np.array(self.game.get_state_1())
        return np.array(self.game.get_state_2())

    def _food_distance(self):
        head = self.game.snake[0]
        return abs(head[0] - self.game.food[0]) + abs(head[1] - self.game.food[1])

    def _calculate_reward(self, done):
        if done:
            return -100  # Death penalty

        # Calculate food reward
        food_reward = 50 * (self.game.score - self.prev_score)
        self.prev_score = self.game.score

        # Calculate distance reward
        curr_dist = self._food_distance()
        distance_reward = 0
        if self.prev_food_distance is not None:
            distance_reward = 2 if curr_dist < self.prev_food_distance else -1
        self.prev_food_distance = curr_dist

        # Time penalty
        time_penalty = -0.2

        return food_reward + distance_reward + time_penalty


# --- Enhanced DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_indices = [0, 1, 2]  # Simplified action representation

        # Hyperparameters
        self.gamma = 0.99  # Increased discount factors
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # Slower decay
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.target_update_freq = 50  # Update target network every 50 episodes

        # Networks and memory
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.memory = deque(maxlen=10000)  # Larger replay buffer

    def _build_model(self):
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_indices)
        return np.argmax(self.model.predict(np.array([state]), verbose=0)[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Batch processing for efficiency
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        # Predict Q-values in batch
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        # Calculate target Q-values
        target_q = current_q.copy()
        batch_index = np.arange(self.batch_size)
        target_q[batch_index, actions] = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)

        # Train on full batch
        self.model.fit(states, target_q, verbose=0, batch_size=self.batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


# --- Enhanced Training Loop ---
if __name__ == "__main__":
    EPISODES = 1000
    env = SnakeEnv(grid_size=20, state_type='get_state_2')  # Use richer state
    agent = DQNAgent(env.state_size, 3)
    scores = []

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done and step < 500:  # Longer episodes
            action_idx = agent.act(state)
            action = env.action_space[action_idx]
            next_state, reward, done = env.step(action)
            agent.remember(state, action_idx, reward, next_state, done)

            # Train more frequently
            if len(agent.memory) > agent.batch_size and step % 4 == 0:
                agent.replay()

            state = next_state
            total_reward += reward
            step += 1

        # Update target network periodically
        if e % agent.target_update_freq == 0:
            agent.update_target_network()

        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])  # Rolling average

        print(f"Ep: {e}/{EPISODES} | "
              f"Score: {total_reward:.1f} | "
              f"Avg: {avg_score:.1f} | "
              f"ε: {agent.epsilon:.3f} | "
              f"Mem: {len(agent.memory)}")

    # Save final model
    agent.model.save("dqn_snake_enhanced.h5")