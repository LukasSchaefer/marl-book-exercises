# Exercises for Barcelona Summer School 2024 -- Multi-Agent Reinforcement Learning

> [!NOTE]  
> For these exercises, we will install Python packages and execute them with Python 3. We recommend using a virtual environment to avoid conflicts with other Python installations, e.g. using `venv` or `conda`. All exercises were tested with Python 3.10.

## Tuesday: Tabular Multi-Agent Reinforcement Learning

In this exercise, we will implement the MARL algorithm independent Q-learning (IQL), in which each agent independently learns its Q-function using Q-learning updates. We will train agents in the matrix game of "Prisoners' Dilemma", regularly evaluate their performance, and visualise their learned value functions.

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


## Wednesday: Actor-Critic Algorithms in Level-Based Foraging

In the following exercises, we will make use of the textbook codebase. This [codebase](https://github.com/marl-book/codebase) has been designed to accompany the MARL textbook and contains easy-to-understand implementations deep MARL algorithms. The algorithm implementations are self-contained and focus on simplicity.

### Multi-Agent Deep Reinforcement Learning via the MARL Textbook Codebase

> [!NOTE]  
> Implementation tricks, while necessary for some algorithms to perform well, are sparse in this codebase as not to make the code very complicated. As a result, some performance has been sacrificed for simplicity.

Make sure you clone the marl-book codebase within this repository as a submodule by running the following command:

```bash
git submodule update --init --recursive
```

After, you should find the `marl-book-codebase` directory within this directory. Navigate to the `marl-book-codebase` directory and install the required dependencies and codebase by running the following command:

```bash
cd marl-book-codebase
pip install -r requirements.txt
pip install -e .
```

You can familiarise yourself with the codebase by reading the `README.md` file within the `marl-book-codebase` directory, and by looking through the files. Generally, you can find the following directories and files:
- `marlbase/run.py`: The main script to start training runs.
- `marlbase/search.py`: The script to initiate multiple training runs, designed to search for hyperparameters.
- `marlbase/ac`: The directory containing the implementation of the actor-critic algorithms advantage actor-critic (A2C) and proximal policy optimisation (PPO). These algorithms can be trained independently or with large centralised critics that are conditioned on the joint observations of all agents.
- `marlbase/dqn`: The directory containing the implementation of the deep Q-network (DQN) algorithm as well as the extensions of value decomposition algorithms value decomposition networks (VDN) and QMIX.
- `marlbase/configs`: The directory contains configuration files that specify hyperparameters for the training runs. The `default.yaml` specifies several default hyperparameters that can be overridden by the user, and all algorithms under `marlbase/configs/algorithm` have further hyperparameters specified in their respective files.

To be able to train agents with the codebase, let's now install the [level-based foraging environment](https://github.com/uoe-agents/lb-foraging) for a range of multi-agent tasks by running the following command:

```bash
pip install git+https://github.com/uoe-agents/lb-foraging.git
```

Now we are ready to train some MARL agents with deep reinforcement learning algorithms! To test this, navigate to the `marlbase` directory and run the following command to train an independent A2C agent in a simple level-based foraging task:

```bash
python run.py +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=100000
```

You should see regular progress of training being printed to the console. For more details on the codebase, please refer to the `README.md` file within the `marl-book-codebase` directory.


### Train Agents with Independent Advantage Actor-Critic (IA2C)

First, we will look at the independent A2C (IA2C) algorithm that is a multi-agent extension of the advantage actor-critic (A2C) algorithm, where each agent learns its policy and value function independently. To start a training run in a level-based foraging task, we can use the following command:

```bash
python run.py +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 algorithm.name=ia2c env.standardise_rewards=True algorithm.video_interval=200000 seed=0
```

This will train the IA2C agents in the "Foraging-8x8-3p-2f-v3" level-based foraging task and periodically evaluate their performance, and generate videos of rollouts. After training, find the corresponding results directory under `outputs/lbforaging:Foraging-8x8-3p-2f-v3/ia2c/...` in which we store logged training metrics in a `results.csv` file, and videos in the `videos` directory.

Watch some of the videos to see how the IA2C agents perform and gradually improve their policies over time. You can also plot the episodic returns throughout training by running the following command:

```bash
python utils/postprocessing/plot_runs.py --source outputs
```

You can also visualise other metrics stored within the `results.csv` file by specifying the respective metric name as an argument to the `plot_runs.py` script. For example, you can plot the entropy of the policies throughout training by running the following command:

```bash
python utils/postprocessing/plot_runs.py --source outputs --metric entropy
```

### Play with Hyperparameters

Try changing some hyperparameters of the algorithm and see how the training is affected! For example, you could change the learning rate, the entropy regularisation coefficient, or the network architecture of the agents. You can run multiple training runs with different hyperparameters in parallel by using multiruns as follows:

```bash
python run.py -m +algorithm=ia2c env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50 algorithm.total_steps=1000000 env.standardise_rewards=True seed=0 algorithm.entropy_coef=0.001,0.01,0.1
```

This will execute three training runs with different entropy regularisation coefficients in parallel. 

> [!NOTE]  
> This will take some time to complete, likely only a few minutes depending on the available hardware.

All results will be stored under `multirun/`. To identify the best hyperparameters or plot episodic returns of all training configurations, you can run the following commands:

```bash
python utils/postprocessing/find_best_hyperparams.py --source multirun
python utils/postprocessing/plot_runs.py --source multirun
```

### Comparison to Centralised Critics

Next, we will compare the performance of the IA2C agents to agents trained with the same fundamental MARL algorithm but instead we will use centralised critics that are conditioned on the joint observations of all agents. As a challenging task, we will train agents in a task of the [robotic warehouse](https://github.com/uoe-agents/robotic-warehouse) environment. To install the environment, run the following command:

```bash
pip install git+https://github.com/uoe-agents/robotic-warehouse.git
```

We could then train IA2C and multi-agent advantage actor-critic (MAA2C) agents in the "rware-tiny-4ag-v2" task by running the following command:

```bash
python run.py -m +algorithm=ia2c,maa2c env.name="rware:rware-tiny-4ag-v2" env.time_limit=500 algorithm.total_steps=20000000 env.standardise_rewards=True seed=0,1,2
```

However, this would take a long time to complete and (depending on the available hardware) might not be feasible since it will run six training runs in parallel for 20,000,000 time steps! Instead, we can use the training data that we have already prepared for this task. You can find the training data for IA2C, MAA2C as well as IPPO and MAPPO in the "rware-tiny-4ag-v2" task in the `deep_marl_data/rware_tiny_4ag` directory. Compare the performance of the algorithms and inspect several metrics such as the episodic returns (`mean_episode_returns`), the entropy of the policies throughout training (`entropy`), or the loss of the actor (`actor_loss`) and critic (`critic_loss`). You can plot these metrics by running the following command:

```bash
python utils/postprocessing/plot_runs.py --source path/to/deep_marl_data/rware_tiny_4ag --metric ...
```

You can also have a look at the videos of the agents' rollouts stored throughout training! These can be found in the `videos` directory of the respective training run, or you can load the checkpoints of models at certain points of training and generate videos of their rollouts by running the following command:

```bash
python eval.py path=path/to/deep_marl_data/rware_tiny-4ag/... env.name="lbforaging:Foraging-8x8-3p-2f-v3" env.time_limit=50
```
with the path pointing to the run directory of the training run you want to evaluate. By default, this will load the last checkpoint of the training run and generate a video of the agents' rollouts. You can also specify the checkpoint to load by adding the `load_step` argument to the command with the respective timestep the checkpoint is from (you can check the `checkpoints` directory of a run to see all available checkpoints).

> [!NOTE]
> We note that only minimal hyperparameter tuning was performed for all of the training runs in the `deep_marl_data` directory so the performance and relative ranking of algorithms might not be reflective of their capabilities under tuned hyperparameters!


## Thursday: Value Decomposition Algorithms

In this exercise, we will take a look at independent DQN (IDQN) and value decomposition algorithms such as value decomposition networks (VDN) and QMIX. These algorithms extend IDQN for common-reward multi-agent tasks and decompose the joint action-value function into individual utility functions for each agent. We will look at results of these algorithms in a level-based foraging task and a task of the SMAClite environment (more on this later).

### Level-Based Foraging with Value Decomposition Algorithms

First, we will take a look at the level-based foraging task `Foraging-8x8-2p-3f` and note that to train IDQN in this task with common rewards, we will use the `CooperativeReward` wrapper that gives each agent the same reward defined by the sum of all individual agents' rewards. To train IDQN with this common reward setup, we can run the following command:

```bash
python run.py +algorithm=idqn env.name="lbforaging:Foraging-8x8-2p-3f-v3" env.time_limit=50 algorithm.total_steps=4000000 algorithm.eval_interval=100000 algorithm.log_interval=100000 env.standardise_rewards=True env.wrappers="[CooperativeReward]"
```

Similarly, we can train VDN and QMIX agents in the same task by substituting the `+algorithm` argument with `vdn` or `qmix`. In this case, we do not explicitly need to specify the `CooperativeReward` wrapper as the VDN and QMIX algorithm configurations already specify this setup.

Since training all algorithms in this task would take notable time, we provide training data for IDQN, VDN, and QMIX agents in the `deep_marl_data/lbf_8x8-2p-3f_coop` directory. We suggest to compare the performance in episodic returns and other metrics of these algorithms. To get a sense of the learned policies, you can view the provided videos of rollouts, or generate rollout videos by loading checkpoints and running the evaluation script as before.

### SMAClite with Value Decomposition Algorithms

Lastly, we will look at the fully cooperative multi-agent environment of [SMAClite](https://github.com/uoe-agents/smaclite). SMAClite closely simulates the [StarCraft multi-agent challenge (SMAC)](https://github.com/oxwhirl/smac) which is a commonly used environment for fully cooperative multi-agent tasks. SMAC tasks represent challenging combat scenarios in StarCraft II in which each agent controls a single unit of a team and the team of agents must defeat an enemy team of units in combat. The enemy team is controlled by a built-in AI that plays against the agents. All agents in SMAC(lite) receive identical rewards and only observe partial information.

You can install the SMAClite environment by running the following command:

```bash
pip install git+https://github.com/uoe-agents/smaclite.git
```

For example, you can start a training run of IDQN in the 2s3z scenario of SMAClite with the following command:

```bash
python run.py +algorithm=idqn env.name="smaclite/2s3z-v0" env.time_limit=150 algorithm.total_steps=1000000 algorithm.eval_interval=1000 algorithm.log_interval=1000 env.standardise_rewards=True
```

We note that SMAClite is significantly more computationally expensive than the level-based foraging task, and training runs will take longer to complete. We provide data of training runs of IDQN, VDN, and QMIX in two SMAClite scenarios in the `deep_marl_data/` directory: `2s3z` and `2s_vs_1sc`. We encourage you to compare the performance of these algorithms in these scenarios, inspect the training data as before, and look at the videos of the agents' rollouts. Can you spot interesting behaviours of the agents as training progresses, such as strategic movement of ranged and melee units, or coordinated attacks on enemy units?

