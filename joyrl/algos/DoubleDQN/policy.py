#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-27 12:00:29
Discription: 
'''
import torch
import torch.nn as nn
import math, random
import numpy as np
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import QNetwork
class Policy(BasePolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.gamma = cfg.gamma  
        # e-greedy parameters
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.target_update = cfg.target_update
        self.sample_count = 0
        self.update_step = 0
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary

    def create_graph(self):
        self.policy_net = QNetwork(self.cfg,self.state_size_list).to(self.device)
        self.target_net = QNetwork(self.cfg,self.state_size_list).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.create_optimizer()

    def sample_action(self, state, **kwargs):
        ''' sample action
        '''
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            action = self.predict_action(state)
        else:
            action = [self.action_space.sample()]
        return action
    
    @torch.no_grad()
    def predict_action(self,state,**kwargs):
        ''' predict action
        '''
        state = [torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)]
        _ = self.policy_net(state)
        actions = self.policy_net.action_layers.get_actions()
        return actions

    def learn(self, **kwargs):
        ''' train policy
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        _ = self.policy_net(states)
        q_values = self.policy_net.action_layers.get_qvalues()
        actual_qvalues = q_values.gather(1, actions.long())

        # compute next Q values Q(s_t+1, a)
        _ = self.policy_net(next_states)
        next_q_values = self.policy_net.action_layers.get_qvalues()

        # compute next target Q values Q'(s_t+1, a)，which is different from DQN
        _ = self.target_net(next_states)
        next_target_qvalues = self.target_net.action_layers.get_qvalues()

        next_target_q_values_action = next_target_qvalues.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))

        expected_q_values = rewards + self.gamma * next_target_q_values_action * (1 - dones)  
        self.loss = nn.MSELoss()(actual_qvalues, expected_q_values)  
        self.optimizer.zero_grad()  
        self.loss.backward()  
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target net every C steps
        if self.update_step % self.target_update == 0: 
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_step += 1
        self.update_summary() # update summary