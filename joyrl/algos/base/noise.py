import numpy as np

class OUNoiseSingleAction(object):
    ''' Ornstein–Uhlenbeck Noise
    '''
    def __init__(self, action_low, action_high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.low = action_low
        self.high = action_high
        self.reset()

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        self.obs = np.ones(1) * self.mu  # reset the noise

    def evolve_obs(self):
        ''' evolove the Ornstein–Uhlenbeck Noise
        '''
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(1)  # Ornstein–Uhlenbeck process
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ''' add noise to action
        '''
        ou_obs = self.evolve_obs()
        #  decay the action noise, as described in the paper
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)[0]  
    
class OUNoise(object):
    ''' Ornstein–Uhlenbeck Noise for multi-head action space
    '''
    def __init__(self, action_size_list, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=10000):
        self.n_action_heads = len(action_size_list)
        self.noise_list = []
        for i in range(self.n_action_heads):
            action_low = action_size_list[i][0]
            action_high = action_size_list[i][1]
            self.noise_list.append(OUNoiseSingleAction(action_low, action_high, mu, theta, max_sigma, min_sigma, decay_period))

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        for i in range(self.n_action_heads):
            self.noise_list[i].reset()
    
    def get_action(self, action, t=0):
        ''' add noise to action
        '''
        action_ = []
        for i in range(self.n_action_heads):
            action_.append(self.noise_list[i].get_action(action[i], t))
        return action_