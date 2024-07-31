#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-01-25 09:58:33
LastEditor: JiangJi
LastEditTime: 2024-07-31 10:19:50
Discription: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.noise import OUNoise
from joyrl.algos.base.network import *

class Policy(BasePolicy):
    def __init__(self, cfg, **kwargs):
        super(Policy, self).__init__(cfg, **kwargs)
        self.gamma = cfg.gamma  
        self.alpha = cfg.alpha
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
        q_values = [actor_outputs[i]['q_value'] for i in range(len(self.action_size_list))]
        values_soft = [ self.alpha * torch.logsumexp(q_value / self.alpha, dim=1, keepdim=True) for q_value in q_values]
        probs = [F.softmax((q_value - value_soft), dim=1) for q_value, value_soft in zip(q_values, values_soft)]
        dists = [torch.distributions.Categorical(probs = prob) for prob in probs]
        actions = [dist.sample().cpu().numpy().item() for dist in dists]
        return actions
    
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
            actual_q_value = actor_outputs[i]['q_value'].gather(1, self.actions[i].long())
            # compute next max q value
            next_q_value = target_actor_outputs[i]['q_value']
            next_v_soft = self.alpha * torch.logsumexp(next_q_value / self.alpha, dim=1, keepdim=True)
            # compute target Q values
            target_q_value = self.rewards + (1 - self.dones) * self.gamma * next_v_soft
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
        self.update_summary() # update summary
 