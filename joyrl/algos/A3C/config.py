#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-06-03 13:37:11
LastEditor: JiangJi
LastEditTime: 2024-06-23 17:49:41
Discription: 
'''

class AlgoConfig(object):
    def __init__(self):
        self.independ_actor = True # whether to use independent actor
        self.action_type_list = "continuous" # continuous action space
        self.gae_lambda = 0.95 # lambda for GAE
        self.gamma = 0.99 # discount factor
        self.lr = 0.0001 # for shared optimizer
        self.actor_lr = 0.0003 # learning rate for actor, must be specified if share_optimizer is False
        self.critic_lr = 0.001 # learning rate for critic, must be specified if share_optimizer is False
        self.critic_loss_coef = 0.001 # critic loss coefficient
        self.entropy_coef = 0.01 # entropy coefficient
        self.batch_size = 256 # ppo train batch size
        self.min_policy = 0 # min value for policy (for discrete action space)
        self.buffer_type = 'REPLAY_QUE'
        self.branch_layers = []
        self.merge_layers = []
        self.actor_branch_layers = []
        self.actor_merge_layers = []
        self.critic_branch_layers = []
        self.critic_merge_layers = []
