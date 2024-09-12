# Exercises for Barcelona Summer School 2024 -- Multi-Agent Reinforcement Learning

## Day 1: Tabular Multi-Agent Reinforcement Learning

In this exercise, we will implement the MARL algorithm of independent Q-learning (IQL), in which each agent independently learns its Q-function using Q-learning updates. We will train agents in the matrix game of "Prisoners' Dilemma", regularly evaluate their performance, and visualise their learned value functions.

Note: We will run the "Prisoners' Dilemma" matrix game as a non-repeated game, where agents play a single round of the game and receive rewards based on their actions. This also makes the game stateless. In the implementation of the matrix game (see `matrix_game.py`), the state/ observation given to agents is always a 0-constant.

All the code for this exercise can be found in the `tabular_marl` directory. To get started, navigate to the `tabular_marl` directory and install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

Next, open the `iql.py` file in the `tabular_marl` directory and implement the missing parts of the `IQL` class. You will need to complete the `act` and `update` methods that implement the epsilon-greedy action selection and update the Q-function for the given experience. Also, feel free to modify the `schedule_hyperparameters` function that schedules `epsilon` as a hyperparameter! Once you have implemented these methods, you can run the training and evaluation loop by executing the following command:

```bash
python train_iql.py
```

This will train the IQL agents in the "Prisoners' Dilemma" matrix game and periodically evaluate their performance. You will see regular evaluation results and once training terminates, you will see a plot of the learned Q-values for each agent!

## Multi-Agent Deep Reinforcement Learning via the MARL Textbook Codebase

To accompany the MARL texdtbook, we have designed a codebase that implements basic and easy-to-understand deep MARL ideas. The algorithms are self-contained and the implementations are focusing on simplicity.

Note: Implementation tricks, while necessary for some algorithms, are sparse as not to make the code very complicated. As a result, some performance has been sacrificed.

Make sure you clone the marl-book codebase within this repository as a submodule by running the following command:

```bash
git submodule update --init --recursive
```

After, you should fine the `marl-book-codebase` directory within this directory. Navigate within the `marl-book-codebase` directory and install the required dependencies and codebase by running the following command:

```bash
pip install -r requirements.txt
pip install -e .
```

You can familiarise yourself with the codebase by reading the `README.md` file within the `marl-book-codebase` directory, and by looking through the files. Generally, you can find the following directories and files:
- `marlbase/run.py`: The main script to start training runs.
- `marlbase/search.py`: The script to initiate multiple training runs, designed to search for hyperparameters.
- `marlbase/ac`: The directory containing the implementation of the advantage actor-critic (A2C) algorithm. This algorithm can be trained independently or with large centralised critics that are conditioned on the joint observations of all agents.
- `marlbase/dqn`: The directory containing the implementation of the deep Q-network (DQN) algorithm. On top of the DQN algorithms, we also provide implementations for the value decomposition algorithms value decomposition networks (VDN) and QMIX.
- `marlbase/configs`: The directory contains configuration files that specify hyperparameters for the training runs. The `default.yaml` specifies several default hyperparameters that can be overridden by the user, and all algorithms under `marlbase/configs/algorithm` have further hyperparameters specified in their respective files.

To be able to train agents with the codebase, let's now install the level-based foraging environment for a range of multi-agent tasks by running the following command:

```bash
pip install git+https://github.com/uoe-agents/lb-foraging.git
```

Now we are ready to train some MARL agents with deep reinforcement learning algorithms! To test this, navigate within the `marlbase` directory and run the following command to train an independent A2C agent in a simple level-based foraging task:

```bash
python run.py +algorithm=ac env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=100000
```

You should see regular progress of training being printed to the console. For more details on the codebase, please refer to the `README.md` file within the `marl-book-codebase` directory.

## Day 2: Independent Actor-Critic (IAC) in Level-Based Foraging
