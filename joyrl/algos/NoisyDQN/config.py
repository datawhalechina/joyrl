#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2024-06-03 13:24:36
Discription: 
'''
class AlgoConfig(object):
    ''' algorithm parameters
    '''
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end to get fixed epsilon, i.e. epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay
        self.gamma = 0.95  # reward discount factor
        self.lr = 0.0001  # learning rate
        self.max_buffer_size = 100000  # replay buffer size
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        self.branch_layers = []
        self.merge_layers = [
            {'layer_type': 'noisy_linear', 'layer_size': [64], 'activation': 'ReLU','std_init': 0.4},
            {'layer_type': 'noisy_linear', 'layer_size': [64], 'activation': 'ReLU','std_init': 0.4},
        ]
