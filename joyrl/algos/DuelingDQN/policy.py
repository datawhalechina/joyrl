#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-11-14 23:50:59
LastEditor: JiangJi
LastEditTime: 2024-06-14 22:49:36
Discription: 
'''
from joyrl.algos.DQN.policy import Policy as DQNPolicy

class Policy(DQNPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
