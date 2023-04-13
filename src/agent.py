import numpy as np
import numpy.ma as ma

from src.distributions import boltzmann1d
from src.gridworld import Gridworld

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
    
    def generate_trajectories(self, world: Gridworld, beta: float, start_pos: int, max_T: int):
        trajs = -1 * np.ones((len(self.agents), max_T, 2))

        for m in range(len(self.agents)):
            s = start_pos
            for t in range(max_T):
                a = self.agents[m].get_action(world, s, beta)
                trajs[m, t] = [world.state_to_idx(s), a]

                s, terminal = world.act(s, a)
                if terminal:
                    break
        
        return trajs
        