#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-12-18 13:15:04
LastEditor: JiangJi
LastEditTime: 2024-12-19 13:24:11
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
        self.max_buffer_size = 100000  # replay buffer size
        self.batch_size = 256  # batch size
        self.target_update = 4  # target network update frequency
        self.distributional = True  # if use distributional dqn
        self.n_atoms = 51  # number of atoms
        self.v_min = -10 # min value
        self.v_max = 10 # max value
        self.enable_soft_update = True # enable soft update
        self.tau_soft_update = 0.005 # soft update tau
        # value network layers config
        # [{'name': 'feature_1', 'layers': [{'layer_type': 'linear', 'layer_size': [256], 'activation': 'relu'}, {'layer_type': 'linear', 'layer_size': [256], 'activation': 'relu'}]}]
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
            # {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
            # {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
        ]
