import numpy as np

class OUNoise(object):
    ''' Ornstein–Uhlenbeck Noise
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.n_actions = action_space.shape[0] 
        self.low = action_space.low  
        self.high = action_space.high 
        self.reset()

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        self.obs = np.ones(self.n_actions) * self.mu  # reset the noise

    def evolve_obs(self):
        ''' evolove the Ornstein–Uhlenbeck Noise
        '''
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)  # Ornstein–Uhlenbeck process
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ''' add noise to action
        '''
        ou_obs = self.evolve_obs()
        #  decay the action noise, as described in the paper
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)  