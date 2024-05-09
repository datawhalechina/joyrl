#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-17 01:08:36
LastEditor: JiangJi
LastEditTime: 2024-05-10 01:00:45
Discription: 
'''
import numpy as np
import scipy
import torch
from joyrl.algos.base.data_handler import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_exp_len(self, exps, max_step: int = 1):
        ''' get exp len
        '''
        exp_len = len(exps)
        if exp_len <= max_step or exps[-1].done:
            exp_len = max(exp_len, 0)
        else:
            exp_len = exp_len - max_step
        return exp_len
    
    def handle_and_add_exps(self, exps):
        exp_len = self.get_exp_len(exps)
        next_value = exps[-1].value
        adv = 0 # advantage
        for t in reversed(range(0, exp_len)):
            delta = exps[t].reward + self.cfg.gamma * next_value * (1 - exps[t].done) - exps[t].value
            adv = delta + self.cfg.gamma * self.cfg.lam * (1 - exps[t].done) * adv
            exps[t].advantage = adv
            exps[t].return_sum = adv + exps[t].value
            next_value = exps[t].value
        exps = exps[:exp_len]
        self.buffer.push(exps)
        
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
    # def _discount_cumsum(self,rewards, dones, gamma):
    #     ''' compute discount cumsum
    #     '''
    #     return scipy.signal.lfilter([1], [1, float(-gamma)*(1-dones)], rewards[::-1], axis=0)[::-1]
    # def _compute_returns(self, rewards, dones, gamma):
    #     returns = np.zeros_like(rewards, dtype=float)
    #     start_idx = 0
    #     for i, done in enumerate(dones):
    #         if done or i == len(dones) - 1:
    #             end_idx = i + 1
    #             segment = rewards[start_idx:end_idx]
    #             discounted_segment = self._discount_cumsum(segment, gamma)
    #             returns[start_idx:end_idx] = discounted_segment
    #             start_idx = i + 1
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
    #     return returns
    
    def _compute_returns(self, rewards, dones):
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        return returns

        