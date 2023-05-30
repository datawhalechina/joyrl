#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-17 01:08:36
LastEditor: JiangJi
LastEditTime: 2023-05-17 13:42:25
Discription: 
'''
import numpy as np
from algos.base.data_handlers import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
    def handle_exps_before_train(self, exps):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([[exp.reward] for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([[exp.done] for exp in exps])
        # continue
        probs = [exp.probs[0] for exp in exps] if hasattr(exps[-1],'probs') else None
        log_probs = [exp.log_probs[0] for exp in exps] if hasattr(exps[-1],'log_probs') else None
        # discrete
        value = [exp.value[0] for exp in exps] if hasattr(exps[-1],'value') else None
        mu = [exp.mu[0] for exp in exps] if hasattr(exps[-1],'mu') else None
        sigma = [exp.sigma[0] for exp in exps] if hasattr(exps[-1],'sigma') else None
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 
                'probs': probs, 'log_probs': log_probs, 'value': value, 'mu': mu, 'sigma': sigma}
        return data