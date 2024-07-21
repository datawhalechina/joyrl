#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-07-21 14:55:45
Discription: 
'''
import torch
import numpy as np
from joyrl.algos.base.buffer import BufferCreator
from joyrl.framework.utils import exec_method
import threading

class BaseDataHandler:
    ''' Basic data handler
    '''
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)()
        self.data_after_train = {}
        
    def _get_exp_len(self, exps, max_step: int = 1):
        ''' get exp len
        '''
        exp_len = len(exps)
        if exp_len <= max_step or exps[-1].done:
            exp_len = max(exp_len, 0)
        else:
            exp_len = exp_len - max_step
        return exp_len
    
    def get_training_data(self):
        ''' get training data
        '''
        exps = self.buffer.sample(sequential=True)
        if exps is not None:
            self._handle_exps_before_train(exps)
            return self.data_after_train
        
    def handle_exps_after_interact(self, exps: list) -> list:
        ''' handle exps after interact
        '''
        return exps
    
    def add_exps(self, exps):
        exps = self.handle_exps_after_interact(exps)
        self.buffer.push(exps)

    def get_training_data(self):
        ''' get training data
        '''
        exps = self.buffer.sample()
        if exps is not None:
            self._handle_exps_before_train(exps)
            return self.data_after_train
    
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps = self.buffer.sample()
        if exps is not None:
            self._handle_exps_before_train(exps)
            return self.data_after_train


    def _handle_exps_before_train(self, exps: list):
        ''' convert exps to training data
        '''
        model_steps = np.array([exp.model_step for exp in exps]) # [batch_size]
        # multi-head state, [[batch_size, state_dim_head_0], [batch_size, state_dim_head_1], ...]
        states = [ np.array([exp.state[i] for exp in exps]) for i in range(len(exps[0].state))]
        next_states = [ np.array([exp.next_state[i] for exp in exps]) for i in range(len(exps[0].next_state))]
        # multi-head action, [[batch_size, action_dim_head_0], [batch_size, action_dim_head_1], ...]
        actions = [ np.array([exp.action[i] for exp in exps]) for i in range(len(exps[0].action))]
        rewards = np.array([exp.reward for exp in exps]) # [batch_size]
        dones = np.array([exp.done for exp in exps]) # [batch_size] 
        self.data_after_train = {'model_steps': model_steps, 'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}

    def handle_exps_after_train(self):
        ''' handle exps after train
        '''
        pass