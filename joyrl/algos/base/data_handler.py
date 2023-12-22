#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2023-12-03 14:43:23
Discription: 
'''
import numpy as np
from joyrl.algos.base.buffers import BufferCreator
from joyrl.algos.base.exps import Exp

class BaseDataHandler:
    ''' Basic data handler
    '''
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)()
        self.data_after_train = {}
    def add_transition(self, transition):
        ''' add transition to buffer
        '''
        exp = self._create_exp(transition)
        self.buffer.push(exp)

    def add_exps(self, exps):
        self.buffer.push(exps)
        
    def add_data_after_learn(self, data):
        ''' add update data
        '''
        self.data_after_train = data
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps = self.buffer.sample()
        if exps is not None:
            return self.handle_exps_before_train(exps)
        else:
            return None
    def _create_exp(self,transtion):
        ''' create experience
        '''
        return [Exp(**transtion)]
    def handle_exps_before_train(self, exps, **kwargs):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
    def handle_exps_after_train(self):
        ''' handle exps after train
        '''
        pass