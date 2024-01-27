#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-27 11:58:27
Discription: 
'''
import torch
import numpy as np
from joyrl.algos.base.buffer import BufferCreator
from joyrl.algos.base.experience import Exp

class BaseDataHandler:
    ''' Basic data handler
    '''
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)()
        self.data_after_train = {}

    def add_exps(self, exps):
        self.buffer.push(exps)
        
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps = self.buffer.sample()
        if exps is not None:
            return self.handle_exps_before_train(exps)
        else:
            return None

    def handle_exps_before_train(self, exps, **kwargs):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        # convert numpy to tensor
        states = torch.tensor(states, device=self.cfg.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.cfg.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.cfg.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        dones = torch.tensor(dones, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=1)
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
    
    def handle_exps_after_train(self):
        ''' handle exps after train
        '''
        pass