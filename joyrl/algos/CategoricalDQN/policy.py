#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-01-25 09:58:33
LastEditor: JiangJi
LastEditTime: 2024-12-19 13:41:36
Discription: 
'''
import torch
import torch.nn as nn
import math,random
import numpy as np
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import *

class Policy(BasePolicy):
    def __init__(self, cfg, **kwargs):
        super(Policy, self).__init__(cfg, **kwargs)
        self.gamma = cfg.gamma  
        # e-greedy parameters
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.target_update = cfg.target_update
        self.sample_count = 0
        self.update_step = 0
        self.atoms = torch.linspace(self.cfg.v_min, self.cfg.v_max, steps=self.cfg.n_atoms)
        self.delta_z = float(self.cfg.v_max - self.cfg.v_min) / (self.cfg.n_atoms - 1)
        self.proj_dist = torch.zeros((self.cfg.batch_size, self.cfg.n_atoms), device=self.device) # [batch_size, n_atoms]
        
    def create_model(self):
        self.model = QNetwork(self.cfg, self.state_size_list).to(self.device)
        self.target_model = QNetwork(self.cfg, self.state_size_list).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict()) # or use this to copy parameters

    def load_model_meta(self, model_meta):
        super().load_model_meta(model_meta)
        if model_meta.get('sample_count') is not None:
            self.sample_count = model_meta['sample_count']

    def sample_action(self, state,  **kwargs):
        ''' sample action
        '''
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        self.update_model_meta({'sample_count': self.sample_count})
        if random.random() > self.epsilon:
            # before update, the network inference time may be longer
            action = self.predict_action(state) 
        else:
            action = get_model_actions(self.model, mode = 'random', actor_outputs = [{}] * len(self.action_size_list))
        return action
    
    @torch.no_grad()
    def predict_action(self,state, **kwargs):
        ''' predict action
        '''
        state = self.process_sample_state(state)
        model_outputs = self.model(state)
        actor_outputs = model_outputs['actor_outputs']
        actions = get_model_actions(self.model, mode = 'predict', actor_outputs = actor_outputs)
        return actions
    
    def _projection_distribution(self, target_q_dist, rewards, dones):
        ''' 用于计算下一时刻的分布
        '''
        with torch.no_grad():
            Tz = rewards + (1-dones) * self.gamma * self.atoms
            Tz = Tz.clamp(min=self.cfg.v_min, max=self.cfg.v_max)
            b = (Tz - self.cfg.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            delta_m_l = (u + (l == u) - b) * target_q_dist  # (batch_size, n_atoms)
            delta_m_u = (b - l) * target_q_dist # (batch_size, n_atoms)
            offset = torch.linspace(0, (self.cfg.batch_size - 1) * self.cfg.n_atoms, self.cfg.batch_size,device=self.device).unsqueeze(-1).long() 
            self.proj_dist *= 0
            self.proj_dist.view(-1).index_add_(0, (l + offset).view(-1), delta_m_l.view(-1))
            self.proj_dist.view(-1).index_add_(0, (u + offset).view(-1), delta_m_u.view(-1))
        # return self.proj_dist
    
    def learn(self, **kwargs):
        ''' learn policy
        '''
        super().learn(**kwargs)
        # compute current Q values
        self.summary_loss = []
        tot_loss = 0
        actor_outputs = self.model(self.states)['actor_outputs']
        target_actor_outputs = self.target_model(self.next_states)['actor_outputs']
        for i in range(len(self.action_size_list)):
            actual_q_dist = actor_outputs[i]['q_dist'].gather(1, self.actions[i].long().unsqueeze(1).expand(self.cfg.batch_size, 1, self.cfg.n_atoms)).squeeze(1) # [batch_size, n_atoms]
            target_q_value = target_actor_outputs[i]['q_value']
            # print(self.actions[i].shape,torch.argmax(target_q_value, dim=1).shape)
            target_q_dist = target_actor_outputs[i]['q_dist'].gather(1, torch.argmax(target_q_value, dim=1).unsqueeze(1).unsqueeze(1).expand(self.cfg.batch_size, 1, self.cfg.n_atoms)).squeeze(1) # [batch_size, n_atoms]
            self._projection_distribution(target_q_dist.detach(), self.rewards, self.dones)
            # compute loss
            loss_i = (-(self.proj_dist* actual_q_dist.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()
            tot_loss += loss_i
            self.summary_loss.append(loss_i.item())
        self.optimizer.zero_grad()
        tot_loss.backward()
        # clip to avoid gradient explosion
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.cfg.enable_soft_update:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.cfg.tau_soft_update * param.data + (1 - self.cfg.tau_soft_update) * target_param.data)
        else:
            # hard update target net every C steps
            if self.update_step % self.target_update == 0: 
                self.target_model.load_state_dict(self.model.state_dict())
        self.update_step += 1
        self.update_summary() # update summary
 