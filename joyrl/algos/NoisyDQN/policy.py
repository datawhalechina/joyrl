#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 11:23:49
LastEditor: JiangJi
LastEditTime: 2024-06-03 13:25:20
Discription: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random

from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import QNetwork

class Policy(BasePolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.gamma = cfg.gamma  
        # e-greedy parameters
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.target_update = cfg.target_update
        self.sample_count = 0
        self.update_step = 0

    def create_model(self):
        self.model = QNetwork(self.cfg, self.state_size_list).to(self.device)
        self.target_model = QNetwork(self.cfg, self.state_size_list).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()) # or use this to copy parameters

    def sample_action(self, state,  **kwargs):
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
    
    def predict_action(self,state, **kwargs):
        ''' predict action
        '''
        state = [torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)]
        model_outputs = self.model(state)
        actor_outputs = model_outputs['actor_outputs']
        actions = self.model.action_layers.get_actions(mode = 'predict', actor_outputs = actor_outputs)
        return actions

    def learn(self, **kwargs):
        ''' train policy
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        self.summary_loss = []
        tot_loss = 0
        actor_outputs = self.model(states)['actor_outputs']
        target_actor_outputs = self.target_model(next_states)['actor_outputs']
        for i in range(len(self.action_size_list)):
            actual_q_value = actor_outputs[i]['q_value'].gather(1, actions[i].long())
            # compute next max q value
            next_q_value_max = target_actor_outputs[i]['q_value'].max(1)[0].unsqueeze(dim=1)
            # compute target Q values
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value_max
            # compute loss
            loss_i = nn.MSELoss()(actual_q_value, target_q_value)
            tot_loss += loss_i
            self.summary_loss.append(loss_i.item())
        self.optimizer.zero_grad()
        tot_loss.backward()
        # clip to avoid gradient explosion
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target net every C steps
        if self.update_step % self.target_update == 0: 
            self.target_model.load_state_dict(self.model.state_dict())
        self.update_step += 1
        self.model.reset_noise()
        self.target_model.reset_noise()
        self.update_summary() # update summary
 
