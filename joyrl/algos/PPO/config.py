#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-20 21:53:39
LastEditor: JiangJi
LastEditTime: 2024-06-11 23:34:11
Discription: 
'''
class AlgoConfig(object):
    def __init__(self):
        self.independ_actor = False # whether to use independent actor
        # whether actor and critic share the same optimizer
        self.ppo_type = 'clip' # clip or kl
        self.eps_clip = 0.2 # clip parameter for PPO
        self.gae_lambda = 0.95 # lambda for GAE
        # for kl penalty version of PPO
        self.kl_target = 0.1 # target KL divergence
        self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
        self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
        self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper
        self.action_type_list = "continuous" # continuous action space
        self.return_form = 'mc' # 'mc' or 'td' or 'gae'
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.lr = 0.0001 # for shared optimizer
        self.actor_lr = 0.0003 # learning rate for actor, must be specified if share_optimizer is False
        self.critic_lr = 0.001 # learning rate for critic, must be specified if share_optimizer is False
        self.critic_loss_coef = 0.001 # critic loss coefficient
        self.entropy_coef = 0.01 # entropy coefficient
        self.batch_size = 256 # ppo train batch size
        self.sgd_batch_size = 32 # sgd batch size
        self.actor_hidden_dim = 256 # hidden dimension for actor
        self.critic_hidden_dim = 256 # hidden dimension for critic
        self.min_policy = 0 # min value for policy (for discrete action space)
        self.buffer_type = 'REPLAY_QUE'
        self.branch_layers = []
        self.merge_layers = []
        self.actor_branch_layers = []
        self.actor_merge_layers = []
        self.critic_branch_layers = []
        self.critic_merge_layers = []
