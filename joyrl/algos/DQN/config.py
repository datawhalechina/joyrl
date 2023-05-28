#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:55:26
LastEditor: JiangJi
LastEditTime: 2023-05-27 20:53:59
Discription: 
'''

class AlgoConfig():
    ''' algorithm parameters
    '''
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end to get fixed epsilon, i.e. epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay
        self.gamma = 0.95  # reward discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_type = 'REPLAY_QUE' # replay buffer type
        self.buffer_size = 100000  # replay buffer size
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        # value network layers config
        self.value_layers = [
            {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
        ]
