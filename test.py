import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=1000, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.memory_size = memory_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.ravel(state)
        next_state = np.ravel(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([np.ravel(np.array(sample[0])) for sample in minibatch])
        actions = np.array([sample[1] for sample in minibatch])
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([np.ravel(np.array(sample[3])) for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])

        targets = self.model.predict(states)
        next_q_values = self.model.predict(next_states)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])

            targets[i, actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=2)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def play_game(agent1, agent2, rounds, rewards_matrix):
    total_rewards = {0: 0, 1: 0}

    for episode in range(rounds):
        state = np.zeros((1, 2))
        action1 = agent1.act(state)
        action2 = agent2.act(state)
        reward1 = rewards_matrix[action1][action2]
        reward2 = rewards_matrix[action2][action1]

        agent1.remember(state, action1, reward1, state, False)
        agent2.remember(state, action2, reward2, state, False)

        total_rewards[0] += reward1
        total_rewards[1] += reward2

    return total_rewards

def train_agents(agent1, agent2, rounds, rewards_matrix):
    rewards_history = []

    for episode in range(rounds):
        total_rewards = play_game(agent1, agent2, 1, rewards_matrix)
        print(f"Episode {episode + 1}, Agent 1 Total Rewards: {total_rewards[0]}, Agent 2 Total Rewards: {total_rewards[1]}")

        agent1.replay(32)
        total_rewards = play_game(agent2, agent1, 1, rewards_matrix)
        print(f"Episode {episode + 1}, Agent 2 Total Rewards: {total_rewards[1]}")
        agent2.replay(32)

        rewards_history.append(total_rewards)

    plot_rewards(rewards_history)

def plot_rewards(rewards_history):
    agent1_rewards = [episode[0] for episode in rewards_history]
    agent2_rewards = [episode[1] for episode in rewards_history]

    plt.plot(agent1_rewards, label='Agent 1')
    plt.plot(agent2_rewards, label='Agent 2')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Training Progress')
    plt.legend()
    plt.show()

def main():
    state_size = 2
    action_size = 2
    agent1 = DQNAgent(state_size, action_size)
    agent2 = DQNAgent(state_size, action_size)

    rounds = 50
    rewards_matrix = np.array([[1, 5], [0, 5]])  # Adjust rewards as needed

    train_agents(agent1, agent2, rounds, rewards_matrix)

    final_rewards = play_game(agent1, agent2, 1, rewards_matrix)
    print("Final Game Results:")
    print(f"Agent 1 Total Rewards: {final_rewards[0]}")
    print(f"Agent 2 Total Rewards: {final_rewards[1]}")

if __name__ == "__main__":
    main()
