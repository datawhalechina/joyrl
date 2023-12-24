import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
import numpy as np
from typing import Optional
import joyrl

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def one_hot(x, n):
    """Create a one-hot encoding of x for n classes.

    Args:
        x: An integer.
        n: Number of classes.

    Returns:
        A numpy array of shape (n,).
    """
    return np.identity(n)[x:x + 1].squeeze()
class CustomCliffWalkingEnv(Env):

    def __init__(self):
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        # use euclidean distance as reward, instead of constant -1
        dist = np.linalg.norm(np.array(new_position) - np.array(terminal_state))
        is_terminated = tuple(new_position) == terminal_state
        if is_terminated:
            return [(1.0, new_state, 10, is_terminated)]
        return [(1.0, new_state, -dist, is_terminated)]

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}
        return one_hot(int(s), self.nS), r, t, False, {"prob": p} # if you want to use one-hot state representation


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        return int(self.s),{"prob": 1.0}
        return one_hot(int(self.s), self.nS), {"prob": 1.0}
    
if __name__ == "__main__":
    my_env = CustomCliffWalkingEnv()
    yaml_path = "../presets/ToyText/CliffWalking-v0/CustomCliffWalking-v0_DQN.yaml"
    joyrl.run(yaml_path = yaml_path, env = my_env)