import numpy as np
import random
from collections import deque

from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Import your SnakeGame from main.
# Ensure that main.py is in the same directory and contains the SnakeGame class.
from main import SnakeGame

# --- Environment Wrapper ---
class SnakeEnv:
    def __init__(self, grid_size=20, state_type='get_state_1'):
        self.game = SnakeGame(w=grid_size, h=grid_size)
        self.action_space = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # three possible actions
        self.state_type = state_type
        if self.state_type == 'get_state_1':
            self.state_size = 11
        elif self.state_type == 'get_state_2':
            self.state_size = 15
        else:
            self.state_size = 11

    def reset(self):
        self.game.reset()
        if self.state_type == 'get_state_1':
            return np.array(self.game.get_state_1())
        else:
            return np.array(self.game.get_state_2())

    def step(self, action):
        self.game.update(action)
        if self.state_type == 'get_state_1':
            next_state = np.array(self.game.get_state_1())
        else:
            next_state = np.array(self.game.get_state_2())
        if self.game.game_over():
            reward = -100  # heavy penalty for dying
            done = True
        else:
            # You might try alternative reward schemes; e.g., reward food more or penalize steps differently.
            reward = 10 * self.game.score - 1
            done = False
        return next_state, reward, done

    def render(self):
        pass  # You can implement rendering if desired.

# --- DQN Agent with Target Network ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Replay memory
        self.memory = deque(maxlen=2000)
        # Discount factor
        self.gamma = 0.95
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Build main Q-network and target network
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        # Update target network every 'target_update_freq' replays
        self.target_update_freq = 10
        self.target_update_counter = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space())
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        action_index = np.argmax(q_values)
        return np.eye(self.action_size)[action_index].tolist()

    def action_space(self):
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Use the target network to compute the Q-value for the next state.
                next_q_values = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target += self.gamma * np.amax(next_q_values)
            target_f = self.model.predict(np.array([state]), verbose=0)
            action_index = np.argmax(action)
            target_f[0][action_index] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        # Decay exploration rate.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

# --- Training Loop ---
if __name__ == "__main__":
    EPISODES = 1000
    env = SnakeEnv(grid_size=20, state_type='get_state_1')
    state_size = env.state_size
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        # You may wish to extend the episode length if needed.
        while not done and step < 200:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            if done:
                print(f"Episode: {e}/{EPISODES} - Steps: {step} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Optionally, save the trained model.
    agent.model.save("dqn_snake_model.h5")
