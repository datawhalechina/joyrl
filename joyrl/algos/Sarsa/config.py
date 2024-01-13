#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-01-11 13:01:13
LastEditor: JiangJi
LastEditTime: 2024-01-11 13:01:13
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
