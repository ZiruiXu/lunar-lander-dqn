import numpy as np
import pickle
from matplotlib import pyplot as plt
import keras
import gymnasium
import os
import PIL

# define the deep Q-network (DQN) agent
class AgentDQN:

    def __init__(self, environment, gamma, learning_rate, start_over=True, batch_size=64, replay_memory_size=200000):
        self.environment = environment
        self.gamma = gamma # discount factor for future rewards
        self.learning_rate = learning_rate

        self.action_space = self.environment.action_space
        self.action_space_size = self.action_space.n
        self.observation_space = self.environment.observation_space
        self.observation_space_size = self.observation_space.shape[0]

        self.batch_size = batch_size # size of the minibatch for training
        self.replay_memory_size = replay_memory_size # size of the memory buffer
        self.replay_memory = [] # memory buffer for experience replay
        self.play_count = 0 # number of steps in total (including all episodes)
        if start_over:
            self.network = self.initialize_network() # random inital network weights
        else:
            self.network = self.load_network("NeuralNetwork.keras") # use saved network
        self.visualize_network("NeuralNetworkVisualization.png") # visualize network
        self.target_network = self.initialize_network()
        self.update_target_network() # let target network = current network
        self.rng = np.random.default_rng(seed=4321)


    # plot the network architecture
    def visualize_network(self, file_name):
        print(self.network.summary())
        keras.utils.plot_model(self.network, to_file=file_name, show_shapes=True, show_dtype=True, show_layer_names=True, show_layer_activations=True, show_trainable=True)

    # initialize dense network with random weights
    def initialize_network(self):
        network = keras.Sequential([
            keras.layers.Dense(512, activation=keras.layers.LeakyReLU(0.1), input_dim=self.observation_space_size),
            keras.layers.Dense(256, activation=keras.layers.LeakyReLU(0.1)),
            keras.layers.Dense(self.action_space_size)
        ])
        network.compile(loss=keras.losses.mean_squared_error,optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return network

    # let target network = current network
    def update_target_network(self):
        self.target_network.set_weights(self.network.get_weights())

    # save current Q-network
    def save_network(self, file_name):
        self.network.save(file_name)

    # load saved Q-network
    def load_network(self, file_name):
        return keras.models.load_model(file_name)

    # epsilon-greedy policy
    def action(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return self.rng.choice(self.action_space_size) # random choice
        else:
            action_values = self.network(state.reshape((1,-1))).numpy()
            return np.argmax(action_values[0]) # select action of highest Q-value

    # store current experience in the memory buffer
    def remember(self, state, action, reward, next_state, termination):
        if self.play_count < self.replay_memory_size: # memory buffer isn't full
            self.replay_memory.append((state, action, reward, next_state, termination))
        else: # memory buffer is full, overwriting the earliest experience
            self.replay_memory[self.play_count%self.replay_memory_size] = (state, action, reward, next_state, termination)
        self.play_count += 1

    # sample experiences from the memory buffer
    def replay_sampling(self):
        random_indices = self.rng.choice(min(self.play_count, self.replay_memory_size)-1, size=self.batch_size-1, replace=False, shuffle=False).tolist()
        random_indices.append(-1)
        random_samples = [self.replay_memory[(self.play_count-2-random_index)%self.replay_memory_size] for random_index in random_indices]
        states = np.array([random_sample[0] for random_sample in random_samples])
        actions = np.array([random_sample[1] for random_sample in random_samples])
        rewards = np.array([random_sample[2] for random_sample in random_samples])
        next_states = np.array([random_sample[3] for random_sample in random_samples])
        terminations = np.array([random_sample[4] for random_sample in random_samples])
        return states, actions, rewards, next_states, terminations

    # update network weights using a minibatch
    def learn(self):
        if  self.play_count >= self.batch_size: # check if there are enough experiences
            states, actions, rewards, next_states, terminations = self.replay_sampling() # sampling experiences
            # compute the target Q-value: Reward + gamma * max_A' Qt(S', A') * (1 - termination)
            TD_targets = rewards + self.gamma * np.max(self.target_network.predict_on_batch(next_states),axis=1) * (1-terminations)
            # compute current Q-value: Q(S, A)
            actions_values = self.network.predict_on_batch(states)
            actions_values[np.arange(self.batch_size), actions] = TD_targets # replace Q(S, A) by the target Q-value
            self.network.train_on_batch(states, actions_values) # one step of gradient descent

    # visualize an episode
    def render(self, episode, step):
        if episode and (episode+1)**(1/2)%5: # only visualize certain episodes
            return
        directory = f"./images/E{episode+1:04d}/"
        if step==-1: # create the folder for the first time
            os.makedirs(directory)
        PIL.Image.fromarray(self.environment.render()).save(directory+f"S{step+1:04d}.png")

    # train the agent
    def train(self, episodes=2000, max_steps_per_episode=1000, reward_scale=0.01):
        total_rewards = [] # store total reward in each episode
        for episode in range(episodes):
            state, _ = self.environment.reset() # initial state
            self.render(episode, -1)
            total_reward = 0
            epsilon=max(0.985**episode,0.01)
            for step in range(max_steps_per_episode):
                action = self.action(state, epsilon=epsilon) # choose an action using the epsilon greedy policy
                next_state, reward, termination, _, _ = self.environment.step(action) # carry out an action
                self.render(episode, step)
                self.remember(state, action, reward*reward_scale, next_state, termination) # store current experience
                total_reward += reward
                state = next_state
                self.learn() # temporal difference (TD) update
                if termination:
                    break
            total_rewards.append(total_reward)

            print("Episode:", episode+1, ".\tTotal rewards in this episode:", total_reward, ".\tStep:", step+1, ".\tepsilon:", epsilon, flush=True)
            if episode < 150 or total_reward > max(total_rewards[-20:-1]):
                self.update_target_network() # Update target network if total reward is high or in early training states
            if (episode+1)%200 == 0: # checkpoint
                plot_rewards(total_rewards, "TrainingRewards.png")
                self.save_network("NeuralNetwork.keras")
        return total_rewards

# plot total reward for each episode
def plot_rewards(rewards, file_name):
    plt.rcParams.update({"font.size": 17})
    plt.scatter(range(1,len(rewards)+1), rewards, s=0.1)
    plt.xlabel("Episodes")
    plt.xlabel("Total reward in each episode")
    plt.savefig(file_name)
    plt.close()


if __name__ == "__main__":
    learning_rate = 0.0001
    gamma = 0.99 # discount factor for future rewards
    training_episodes = 10000
    reward_scale = 1 # rescale reward if necessary

    # initialize lunar lander environment
    environment = gymnasium.make("LunarLander-v2", render_mode="rgb_array")
    environment.reset(seed=1234)

    # initialize and train the DQN agent
    agent = AgentDQN(environment, gamma, learning_rate, start_over=True)
    training_rewards = agent.train(episodes=training_episodes, reward_scale=reward_scale)

    plot_rewards(training_rewards, "TrainingRewards.png")
    print("Training complete.")
