#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-07-28 11:07:00
Discription: 
'''
import random
import numpy as np

class BaseNoise:
    def __init__(self) -> None:
        pass
    def reset(self):
        pass

class OUNoise(BaseNoise):
    ''' Ornstein–Uhlenbeck Noise
    '''
    def __init__(self, action_low, action_high, **kwargs):
        #mu=0.0, theta=0.15, sigma_max=0.3, sigma_min=0.3, decay_period=10000):
        super(OUNoise, self).__init__()
        self.low = action_low
        self.high = action_high
        self.mu = kwargs.get('mu', 0.0)
        self.theta = kwargs.get('theta', 0.15)
        self.sigma_max = kwargs.get('sigma_max', 0.3)
        self.sigma_min = kwargs.get('sigma_min', 0.3)
        self.sigma = self.sigma_max
        self.decay_period = kwargs.get('decay_period', 10000)
        self.reset()

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        self.obs = self.mu  # reset the noise

    def _evolve_obs(self):
        ''' evolove the Ornstein–Uhlenbeck Noise
        '''
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * random.gauss(0, 1)  # Ornstein–Uhlenbeck process
        self.obs = x + dx
        return self.obs

    def get_action(self, action, **kwargs):
        ''' add noise to action
        '''
        t = kwargs.get('t', 0)
        ou_obs = self._evolve_obs()
        #  decay the action noise, as described in the paper
        self.sigma = self.sigma_max - (self.sigma_max - self.sigma_min) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high) 

class RandomNoise(BaseNoise):
    ''' Ornstein–Uhlenbeck Noise
    '''
    def __init__(self, action_low, action_high, **kwargs):
        super(RandomNoise, self).__init__()
        self.low = action_low
        self.high = action_high
        self.theta = kwargs.get('theta', 0.1)

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        pass

    def get_action(self, action, **kwargs):
        ''' add noise to action
        '''
        delta = self.theta * random.gauss(0, (self.high - self.low)/2)
        return np.clip(action + delta, self.low, self.high)
    
NOISE_DICT = {
        "OU": OUNoise,
        "RANDOM": RandomNoise,
}

class MultiHeadActionNoise(object):
    ''' Ornstein–Uhlenbeck Noise for multi-head action space
    '''
    def __init__(self, noise_type, action_size_list, **kwargs):
        noise_cls = NOISE_DICT.get(noise_type.upper(), None)
        if noise_cls is None:
            raise ValueError("noise_type must be one of {}".format(NOISE_DICT.keys()))
        self.n_action_heads = len(action_size_list)
        self.noise_list = []
        for i in range(self.n_action_heads):
            action_low = action_size_list[i][0]
            action_high = action_size_list[i][1]
            self.noise_list.append(noise_cls(action_low, action_high, **kwargs))

    def reset(self):
        ''' reset the Ornstein–Uhlenbeck Noise
        '''
        for i in range(self.n_action_heads):
            self.noise_list[i].reset()
    
    def get_action(self, action, **kwargs):
        ''' add noise to action
        '''
        action_ = []
        for i in range(self.n_action_heads):
            action_.append(self.noise_list[i].get_action(action[i], **kwargs))
        return action_