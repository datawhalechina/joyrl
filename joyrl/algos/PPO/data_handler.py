#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-17 01:08:36
LastEditor: JiangJi
LastEditTime: 2024-01-27 18:30:42
Discription: 
'''
import numpy as np
import scipy
import torch
from joyrl.algos.base.data_handler import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps = self.buffer.sample(sequential=True)
        if exps is not None:
            return self.handle_exps_before_train(exps)
        else:
            return None
        
    def handle_exps_before_train(self, exps):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        log_probs = [exp.log_prob for exp in exps] 
        returns = self._compute_returns(rewards, dones, self.cfg.gamma).copy()

        actions = torch.tensor(actions, device=self.cfg.device, dtype=torch.float32)
        states = torch.tensor(states, device=self.cfg.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.cfg.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        log_probs = torch.tensor(log_probs, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = torch.tensor(returns, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        # states = np.array([exp.state for exp in exps])
        # actions = np.array([exp.action for exp in exps])
        # rewards = np.array([exp.reward for exp in exps])
        # next_states = np.array([exp.next_state for exp in exps])
        # dones = np.array([exp.done for exp in exps])
        # # continue
        # probs = [exp.probs[0] for exp in exps] if hasattr(exps[-1],'probs') else None
        # log_probs = [exp.log_prob for exp in exps] if hasattr(exps[-1],'log_prob') else None

        # # discrete
        # value = [exp.value[0] for exp in exps] if hasattr(exps[-1],'value') else None
        # mu = [exp.mu[0] for exp in exps] if hasattr(exps[-1],'mu') else None
        # sigma = [exp.sigma[0] for exp in exps] if hasattr(exps[-1],'sigma') else None
        # returns = self._compute_returns(rewards, dones, self.cfg.gamma).copy()
        # data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 
        #         'probs': probs, 'log_probs': log_probs, 'value': value, 'mu': mu, 'sigma': sigma, 'returns': returns}
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 'log_probs': log_probs, 'returns': returns}
        return data
    # def _compute_returns(self, rewards, dones, next_value, use_gae, gamma, tau):
    #     ''' compute returns
    #     '''
    #     if use_gae:
    #         values = np.append(rewards, next_value)
    #         deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    #         gaes = self._discount_cumsum(deltas, gamma * tau * (1 - dones))
    #         returns = gaes + values[:-1]
    #     else:
    #         returns = self._discount_cumsum(rewards, gamma)[:-1]
    #     return returns
    def _discount_cumsum(self,rewards, dones, gamma):
        ''' compute discount cumsum
        '''
        return scipy.signal.lfilter([1], [1, float(-gamma)*(1-dones)], rewards[::-1], axis=0)[::-1]
    def _compute_returns(self, rewards, dones, gamma):
        returns = np.zeros_like(rewards, dtype=float)
        start_idx = 0
        for i, done in enumerate(dones):
            if done or i == len(dones) - 1:
                end_idx = i + 1
                segment = rewards[start_idx:end_idx]
                discounted_segment = self._discount_cumsum(segment, gamma)
                returns[start_idx:end_idx] = discounted_segment
                start_idx = i + 1
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        return returns
    # def _compute_returns(self, rewards, dones):
    #     # monte carlo estimate of state rewards
    #     returns = []
    #     discounted_sum = 0
    #     for reward, done in zip(reversed(rewards), reversed(dones)):
    #         if done:
    #             discounted_sum = 0
    #         discounted_sum = reward + (self.gamma * discounted_sum)
    #         returns.insert(0, discounted_sum)
    #     # Normalizing the rewards:
    #     returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
    #     return returns
    def _discount_cumsum(self, x, discount):
        ''' compute discount cumsum
        '''
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        