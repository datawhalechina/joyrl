#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-07-20 14:15:24
LastEditor: JiangJi
LastEditTime: 2024-07-21 14:52:51
Discription: 
'''
import copy
import torch.nn as nn
from joyrl.algos.base.network import CriticNetwork, ActorNetwork


class Model(nn.Module):
    def __init__(self, cfg ):
        super(Model, self).__init__()
        state_size_list = cfg.obs_space_info.size
        action_size_list = cfg.action_space_info.size
        critic_input_size_list = state_size_list+ [[None, len(action_size_list)]]
        self.actor = ActorNetwork(cfg, input_size_list = state_size_list)
        self.critic = CriticNetwork(cfg, input_size_list = critic_input_size_list)