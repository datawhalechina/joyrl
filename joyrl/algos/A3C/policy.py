#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-07-20 15:55:32
Discription: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
from joyrl.algos.base.network import *
from joyrl.algos.base.policy import BasePolicy
from joyrl.framework.config import MergedConfig

class Policy(BasePolicy):
    def __init__(self, cfg: MergedConfig) -> None:
        super(Policy, self).__init__(cfg)
        self.gamma = cfg.gamma
        self.critic_loss_coef = cfg.critic_loss_coef
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.create_model()
        self.create_optimizer()
        self.create_summary()
    
    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'actor_loss': 0.0,
                'critic_loss': 0.0,
            },
        }
        
    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        if hasattr(self, 'tot_loss'):    
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        self.summary['scalar']['actor_loss'] = self.actor_loss.item()
        self.summary['scalar']['critic_loss'] = self.critic_loss.item()

    def create_model(self):
        self.model = ActorCriticNetwork(self.cfg, self.state_size_list).to(self.device)

    def create_optimizer(self):
        if getattr(self.cfg, 'independ_actor', False):
            self.optimizer = optim.Adam([{'params': self.model.actor.parameters(), 'lr': self.cfg.actor_lr},
                                         {'params': self.model.critic.parameters(), 'lr': self.cfg.critic_lr}])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.cfg.lr)  

    def update_policy_transition(self):
        self.policy_transition = {'value': self.value.detach().cpu().numpy().item(), 'log_prob': self.log_prob}

    def sample_action(self, state, **kwargs):
        state = self.process_sample_state(state)
        model_outputs = self.model(state)
        self.value = model_outputs['value']
        actor_outputs = model_outputs['actor_outputs']
        actions, self.log_prob = get_model_actions_and_log_probs(self.model, mode = 'sample', actor_outputs = actor_outputs)
        self.update_policy_transition()
        return actions

    @torch.no_grad()
    def predict_action(self, state, **kwargs):
        state = self.process_sample_state(state)
        model_outputs = self.model(state)
        actor_outputs = model_outputs['actor_outputs']
        actions = get_model_actions(self.model, mode = 'predict', actor_outputs = actor_outputs)
        return actions
    
    def prepare_data_before_learn(self, **kwargs):
        super().prepare_data_before_learn(**kwargs)
        self.returns = torch.tensor(kwargs.get('returns'), dtype = torch.float32, device = self.device).unsqueeze(dim=1)
        self.values = kwargs.get('values')

    def learn(self, **kwargs):
        super().learn(**kwargs)
        model_outputs = self.model(self.states)
        values = model_outputs['value']
        advantages = self.returns - values.detach() # shape:[batch_size,1]
        values = model_outputs['value']
        actor_outputs = model_outputs['actor_outputs']
        log_probs = get_model_log_probs_action(self.model, actor_outputs, self.actions)
        self.actor_loss = - torch.mean(log_probs * advantages.detach())
        self.critic_loss = self.critic_loss_coef * nn.MSELoss()(self.returns, values) # shape: [batch_size, 1]
        self.tot_loss = self.actor_loss + self.critic_loss + self.entropy_coef * get_model_mean_entropy(self.model, model_outputs['actor_outputs'])
        self.optimizer.zero_grad()
        self.tot_loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_summary()
