# Exercises for Barcelona Summer School 2024 -- Multi-Agent Reinforcement Learning

## Day 1: Tabular Multi-Agent Reinforcement Learning

In this exercise, we will implement the MARL algorithm Independent Q-learning (IQL), in which each agent independently learns its Q-function using Q-learning updates. We will train agents in the matrix game of "Prisoners' Dilemma", regularly evaluate their performance, and visualise their learned value functions.

> [!NOTE]  
> We will run the "Prisoners' Dilemma" matrix game as a non-repeated game, where agents play a single round of the game and receive rewards based on their actions. This also makes the game stateless. In the implementation of the matrix game (see `matrix_game.py`), the state/ observation given to agents is always a 0-constant.

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

To accompany the MARL textbook, we have designed a [codebase](https://github.com/marl-book/codebase) that implements basic and easy-to-understand deep MARL ideas. The algorithms are self-contained and the implementations are focusing on simplicity.

Note: Implementation tricks, while necessary for some algorithms, are sparse as not to make the code very complicated. As a result, some performance has been sacrificed.

Make sure you clone the marl-book codebase within this repository as a submodule by running the following command:

```bash
git submodule update --init --recursive
```

After, you should find the `marl-book-codebase` directory within this directory. Navigate within the `marl-book-codebase` directory and install the required dependencies and codebase by running the following command:

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

To be able to train agents with the codebase, let's now install the [level-based foraging environment](https://github.com/uoe-agents/lb-foraging) for a range of multi-agent tasks by running the following command:

```bash
pip install git+https://github.com/uoe-agents/lb-foraging.git
```

Now we are ready to train some MARL agents with deep reinforcement learning algorithms! To test this, navigate within the `marlbase` directory and run the following command to train an independent A2C agent in a simple level-based foraging task:

```bash
python run.py +algorithm=ac env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=100000
```

You should see regular progress of training being printed to the console. For more details on the codebase, please refer to the `README.md` file within the `marl-book-codebase` directory.

## Day 2: Actor-Critic Algorithms in Level-Based Foraging

In this exercise, we will train some agents in a level-based foraging task using the independent advantage actor-critic (IA2C) algorithm. The IAC algorithm is a multi-agent extension of the advantage actor-critic (A2C) algorithm, where each agent learns its policy and value function independently. 

### Train Agents with Independent Advantage Actor-Critic (IA2C)

Start a training run with the following command:

```bash
python run.py +algorithm=ac env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 algorithm.name="ia2c" algorithm.standardise_rewards=True algorithm.video_interval=10000 seed=0
```

This will train the IA2C agents in the "Foraging-8x8-3p-2f-v3" level-based foraging task and periodically evaluate their performance, and generate videos of rollouts. After training, find the corresponding results directory under `outputs/lbforaging:Foraging-8x8-3p-2f-v3/ia2c/...` in which we store logged training metrics in a `results.csv` file, and videos in the `videos` directory.

Watch some of the videos to see how the IA2C agents perform and gradually improve their policies over time. You can also plot the episodic returns throughout training by running the following command:

```bash
python utils/postprocessing/export_multirun.py --folder outputs/... --export-file myfile.hd5
python utils/postprocessing/plot_best_runs.py --exported-file myfile.hd5
```

### Play with Hyperparameters

Try changing some hyperparameters of the algorithm and see how the training is affected! For example, you could change the learning rate, the entropy regularisation coefficient, or the network architecture of the agents. You can run multiple training runs with different hyperparameters in parallel by using multiruns as follows:

```bash
python run.py -m +algorithm=ac env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 algorithm.name="ia2c" algorithm.standardise_rewards=True seed=0 algorithm.entropy_coef=0.001,0.01,0.1
```

This will execute three training runs with different entropy regularisation coefficients. All results will be stored under `multirun/`. To identify the best hyperparameters, you can run the following commands:

```bash
python utils/postprocessing/export_multirun.py --folder multirun/... --export-file myfile.hd5
python utils/postprocessing/find_best_hyperparams.py --exported-file myfile.hd5
```

### Comparison to Centralised Critics

Next, we will compare the performance of the IA2C agents to agents trained with the same fundamental MARL algorithm but instead we will use centralised critics that are conditioned on the joint observations of all agents. We already prepared a configuration file that will run experiments with both IA2C and MAA2C (IA2C with a centralised critic) algorithms in a more challenging level-based foraging task that requires more cooperation between four agents. You can start the training runs with the following command:

```bash
python search.py run --config configs/sweeps/ia2c_maa2c.yaml --seeds 3 locally --cpus 3
```

> [!NOTE]  
> This command will train both IA2C and MAA2C agents with three different random seeds each with each individual training run training agents for 5,000,000 steps due to the more challenging task. Note that this will take a while to complete!

To adjust the number of random seeds each algorithm is trained with, you can change the `--seeds` argument. The `--cpus` argument specifies the number of runs executed in parallel. After training, you can compare the performance of the IA2C and MAA2C agents by running the following command:

```bash
python utils/postprocessing/export_multirun.py --folder multirun/... --export-file myfile.hd5
python utils/postprocessing/plot_best_runs.py --exported-file myfile.hd5
```


## Day 3: Value Decomposition Algorithm in SMAClite

In this exercise, we will look at the fully cooperative multi-agent tasks of [SMAClite](https://github.com/uoe-agents/smaclite). SMAClite closely simulates the [StarCraft multi-agent challenge (SMAC)](https://github.com/oxwhirl/smac) which is a commonly used environment for fully cooperative multi-agent tasks. SMAC tasks represent challenging combat scenarios in StarCraft II in which each agent controls a single unit of a team and the team of agents must defeat an enemy team of units in combat. The enemy team is controlled by a built-in AI that plays against the agents. All agents in SMAC(lite) receive identical rewards and only observe partial information.

### Train Agents with Independent Deep Q-Network (IDQN)

### Train Agents with QMIX
