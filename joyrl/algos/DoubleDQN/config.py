#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-03-15 22:04:42
LastEditor: JiangJi
LastEditTime: 2024-01-26 10:14:47
Discription: 
'''
class AlgoConfig(object):
    ''' algorithm parameters
    '''
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.99  # discount factor
        self.lr = 0.0001  # learning rate
        self.max_buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        # value network layers config
        self.branch_layers = [
        #     {
        #         'name': 'feature_1',
        #         'layers': 
        #         [
        #             {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        #             {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        #         ]
        #     },
        #     {
        #         'name': 'feature_2',
        #         'layers': 
        #         [
        #             {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        #             {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        #         ]
        #     }
        ]
        self.merge_layers = [
            {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
            {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
        ]

