#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 11:23:49
LastEditor: JiangJi
LastEditTime: 2024-06-14 22:53:19
Discription: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random

from joyrl.algos.DQN.policy import Policy as DQNPolicy


class Policy(DQNPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)


    def learn(self, **kwargs):
        ''' train policy
        '''
        self.prepare_data_before_learn(**kwargs)
        self.summary_loss = []
        tot_loss = 0
        actor_outputs = self.model(self.states)['actor_outputs']
        target_actor_outputs = self.target_model(self.next_states)['actor_outputs']
        for i in range(len(self.action_size_list)):
            actual_q_value = actor_outputs[i]['q_value'].gather(1, self.actions[i].long())
            # compute next max q value
            next_q_value_max = target_actor_outputs[i]['q_value'].max(1)[0].unsqueeze(dim=1)
            # compute target Q values
            target_q_value = self.rewards + (1 - self.dones) * self.gamma * next_q_value_max
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
 
