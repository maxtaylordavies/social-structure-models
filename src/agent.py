import numpy as np

from src.distributions import boltzmann1d
from src.gridworld import Gridworld
from src.utils import random_partition

class Agent:
    def __init__(self, r):
        self.r = r
        self.Q = None

    def get_action(self, world: Gridworld, s: np.ndarray, beta: float):
        if self.Q is None:
            self.Q = world.solve_q_values(rewards=self.r)

        q = self.Q[world.state_to_idx(s)]
        p = boltzmann1d(q / np.sum(q), beta)
        return np.random.choice(len(p), p=p)
    
    def reset(self):
        self.Q = None


class Population:
    def __init__(self, assignments, group_rewards):
        self.z = assignments
        self.R = group_rewards

        self.agents = []
        for i in range(len(self.z)):
            self.agents.append(Agent(self.R[self.z[i]]))


    def generate_trajectories(self, world: Gridworld, beta: float, start_pos: int, T: int):
        trajectories = np.zeros((len(self.agents), T))

        for i in range(len(self.agents)):
            s = start_pos
            for t in range(T):
                a = self.agents[i].get_action(world, s, beta)
                s, _ = world.act(s, a)
                trajectories[i, t] = a

        return trajectories

        