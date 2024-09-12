import copy
import random

import gymnasium as gym
import numpy as np

from iql import IQL
from utils import visualise_q_tables, visualise_q_convergence, visualise_evaluation_returns
from matrix_game import create_pd_game


CONFIG = {
    "seed": 0,
    "gamma": 0.99,
    "total_eps": 20000,
    "ep_length": 1,
    "eval_freq": 400,
    "lr": 0.05,
    "init_epsilon": 0.9,
}


def iql_eval(env, config, q_tables, eval_episodes=500, output=True):
    """
    Evaluate configuration of independent Q-learning on given environment when initialised with given Q-table

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param q_tables (List[Dict[Act, float]]): Q-tables mapping actions to Q-values for each agent
    :param eval_episodes (int): number of evaluation episodes
    :param output (bool): flag whether mean evaluation performance should be printed
    :return (float, float): mean and standard deviation of returns received over episodes
    """
    eval_agents = IQL(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=0.0,
        )
    eval_agents.q_tables = q_tables

    episodic_returns = []
    for _ in range(eval_episodes):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            actions = eval_agents.act(obss)
            obss, rewards, done, _, _ = env.step(actions)
            episodic_return += rewards

        episodic_returns.append(episodic_return)

    mean_return = np.mean(episodic_returns, axis=0)
    std_return = np.std(episodic_returns, axis=0)

    if output:
        print("EVALUATION RETURNS:")
        print(f"\tAgent 1: {mean_return[0]:.2f} ± {std_return[0]:.2f}")
        print(f"\tAgent 2: {mean_return[1]:.2f} ± {std_return[1]:.2f}")
    return mean_return, std_return



def train(env, config, output=True):
    """
    Train and evaluate independent Q-learning in env with provided hyperparameters

    :param env (gym.Env): environment to execute evaluation on
    :param config (Dict[str, float]): configuration dictionary containing hyperparameters
    :param output (bool): flag if mean evaluation results should be printed
    :return (List[List[float]], List[List[float]], List[Dict[Act, float]]):
    """
    agents = IQL(
            num_agents=env.n_agents,
            action_spaces=env.action_space,
            gamma=config["gamma"],
            learning_rate=config["lr"],
            epsilon=config["init_epsilon"],
        )

    step_counter = 0
    max_steps = config["total_eps"] * config["ep_length"]

    evaluation_return_means = []
    evaluation_return_stds = []
    evaluation_q_tables = []

    for eps_num in range(config["total_eps"]):
        obss, _ = env.reset()
        episodic_return = np.zeros(env.n_agents)
        done = False

        while not done:
            agents.schedule_hyperparameters(step_counter, max_steps)
            acts = agents.act(obss)
            n_obss, rewards, done, _, _ = env.step(acts)
            agents.learn(obss, acts, rewards, n_obss, done)

            step_counter += 1
            episodic_return += rewards
            obss = n_obss

        if eps_num > 0 and eps_num % config["eval_freq"] == 0:
            mean_return, std_return = iql_eval(
                env, config, agents.q_tables, output=output
            )
            evaluation_return_means.append(mean_return)
            evaluation_return_stds.append(std_return)
            evaluation_q_tables.append(copy.deepcopy(agents.q_tables))

    return evaluation_return_means, evaluation_return_stds, evaluation_q_tables, agents.q_tables


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    env = create_pd_game()
    # env = gym.make("lbforaging:Foraging-5x5-2p-1f-v3")
    evaluation_return_means, evaluation_return_stds, eval_q_tables, q_tables = train(env, CONFIG)
    visualise_q_tables(q_tables)
    visualise_evaluation_returns(evaluation_return_means, evaluation_return_stds)
    visualise_q_convergence(eval_q_tables, env)
