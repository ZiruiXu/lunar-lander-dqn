# Deep Q-Learning for Lunar Lander

This project implements a deep Q-network (DQN) to solve the Lunar Lander environment from OpenAI Gym. The goal is to train an agent that can successfully land the lunar lander on the moon

## Key Features

* **Deep Q-Network:** Utilizes a neural network to approximate the Q-values for each state-action pair
* **Experience Replay:** Stores past experiences in a replay buffer and samples them randomly for training, improving stability and efficiency
* **Target Network:** Maintains a separate target network for calculating target Q-values, reducing instability during training
* **Epsilon-Greedy Exploration:** Balances exploration and exploitation by gradually decreasing the probability of taking random actions

## Requirements

* Python 3
* Keras
* OpenAI Gym

## More Information

**Check out my personal website for more details:**
[https://ziruixu.github.io/blog/lunar-lander-dqn](https://ziruixu.github.io/blog/lunar-lander-dqn)