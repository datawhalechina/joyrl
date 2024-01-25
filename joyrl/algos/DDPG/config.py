#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-15 13:16:24
LastEditor: JiangJi
LastEditTime: 2024-01-25 12:01:27
Discription: 
'''
import numpy as np
class AlgoConfig:
    def __init__(self):
        self.action_type_list = 'dpg' # action type, dpg: deterministic policy gradient
        self.buffer_type = 'REPLAY_QUE' # replay buffer type
        self.max_buffer_size = 100000  # replay buffer size
        self.batch_size = 128  # batch size
        self.gamma = 0.99  # discount factor
        self.policy_loss_weight = 0.002  # policy loss weight
        self.critic_lr = 1e-3  # learning rate of critic
        self.actor_lr = 1e-4  # learning rate of actor
        self.tau = 0.001  # soft update parameter
        self.value_min = -np.inf  # clip min critic value
        self.value_max = np.inf  # clip max critic value
        # self.actor_layers = [
        #     {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        #     {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        # ]
        # self.critic_layers = [
        #     {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        #     {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        # ]
        self.branch_layers = []
        self.merge_layers = []
        self.actor_branch_layers = []
        self.actor_merge_layers = []
        self.critic_branch_layers = []
        self.critic_merge_layers = []