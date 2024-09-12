from collections.abc import Iterable

import numpy as np
import gymnasium as gym


class MatrixGame(gym.Env):
    def __init__(self, payoff_matrix, ep_length):
        """
        Create matrix game
        :param payoff_matrix: np.array of shape (n_actions_1, n_actions_2, 2)
        :param ep_length: length of episode (before done is True)
        """
        self.payoff = payoff_matrix
        self.n_agents = 2
        n_actions_1, n_actions_2, _ = payoff_matrix.shape
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(1), gym.spaces.Discrete(1)])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(n_actions_1), gym.spaces.Discrete(n_actions_2)])
        self.ep_length = ep_length
        self.last_actions = None

        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return [0] * self.n_agents, {}

    def step(self, actions):
        assert len(actions) == self.n_agents, f"Expected {self.n_agents} actions, got {len(actions)}"
        self.t += 1
        self.last_actions = actions
        rewards = self.payoff[actions[0], actions[1]]

        if self.t >= self.ep_length:
            done = True
        else:
            done = False

        return [0] * self.n_agents, rewards, done, False, {}

    def render(self):
        print(f"Step {self.t} - actions: {self.last_actions}")


def create_pd_game(ep_length=1):
    payoff_matrix = np.array([[[3, 3], [0, 5]], [[5, 0], [1, 1]]])
    return MatrixGame(payoff_matrix, ep_length)

