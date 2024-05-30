#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-24 19:13:11
LastEditor: JiangJi
LastEditTime: 2024-01-13 16:03:27
Discription: 
'''
import numpy as np
from joyrl.algos.base.data_handler import BaseDataHandler


class DataHandler(BaseDataHandler):
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.buffer = []
        self.data_after_train = {}
    def add_exps(self, exps):
        ''' add transition to buffer
        '''
        self.buffer.append(exps)

    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        if len(self.buffer) == 0:
            return None
        exp = self.buffer.pop()[0]
        return self._handle_exps_before_train(exp)
    def _handle_exps_before_train(self, exp, **kwargs):
        ''' convert exps to training data
        '''
        state = np.array(exp.state)
        action = np.array(exp.action)
        reward = np.array(exp.reward)
        next_state = np.array(exp.next_state)
        done = np.array(exp.done)
        data = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
        return data
    def handle_exps_after_train(self):
        ''' handle exps after train
        '''
        pass
    