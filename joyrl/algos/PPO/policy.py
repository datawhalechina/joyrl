#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-07-21 15:16:46
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
        self.ppo_type = 'clip' # clip or kl
        if self.ppo_type == 'kl':
            self.kl_target = cfg.kl_target 
            self.kl_lambda = cfg.kl_lambda 
            self.kl_beta = cfg.kl_beta
            self.kl_alpha = cfg.kl_alpha
        self.gamma = cfg.gamma
        self.critic_loss_coef = cfg.critic_loss_coef
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sgd_batch_size = cfg.sgd_batch_size
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
        if hasattr(self, 'tot_loss') and len(self.tot_losses_epoch) > 0:    
            self.summary['scalar']['tot_loss'] = np.mean(self.tot_losses_epoch)
        self.summary['scalar']['actor_loss'] = np.mean(self.actor_losses_epoch)
        self.summary['scalar']['critic_loss'] = np.mean(self.critic_losses_epoch)

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
        log_probs, returns = kwargs.get('log_probs'), kwargs.get('returns')
        self.log_probs = torch.tensor(log_probs, dtype = torch.float32, device = self.device).unsqueeze(dim=1)
        # self.log_probs = torch.cat(log_probs, dim=0).detach() # [batch_size,1]
        self.returns = torch.tensor(returns, dtype = torch.float32, device = self.device).unsqueeze(dim=1)

    def learn(self, **kwargs):
        super().learn(**kwargs)
        torch_dataset = Data.TensorDataset(*self.states, *self.actions, self.log_probs, self.returns)
        train_loader = Data.DataLoader(dataset = torch_dataset, batch_size = self.sgd_batch_size, shuffle = True)
        self.actor_losses_epoch, self.critic_losses_epoch, self.tot_losses_epoch = [], [], []
        for _ in range(self.k_epochs):
            for data in train_loader:
                # multi-head state
                old_states = []
                for i in range(len(self.states)):
                    old_states.append(data[i])
                idx = len(self.states)
                # multi-head action
                old_actions = []
                for i in range(len(self.actions)):
                    old_actions.append(data[idx+i])
                idx += len(self.actions)
                old_log_probs = data[idx]
                returns = data[idx+1]
                model_outputs = self.model(old_states)
                values = model_outputs['value']
                actor_outputs = model_outputs['actor_outputs']
                new_log_probs = get_model_log_probs_action(self.model, actor_outputs, old_actions)
                # new_log_probs = self.model.action_layers.get_log_probs_action(old_actions)
                entropy_mean = get_model_mean_entropy(self.model, actor_outputs)
                advantages = returns - values.detach() # shape:[batch_size,1]
                # get action probabilities
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_log_probs - old_log_probs) # shape: [batch_size, 1]
                # compute surrogate loss
                surr1 = ratio * advantages # shape: [batch_size, 1]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # compute actor loss
                self.actor_loss = - (torch.mean(torch.min(surr1, surr2)) )
                # compute critic loss
                self.critic_loss = self.cfg.critic_loss_coef * nn.MSELoss()(returns, values) # shape: [batch_size, 1]
                self.actor_losses_epoch.append(self.actor_loss.item())
                self.critic_losses_epoch.append(self.critic_loss.item())
                self.tot_loss = self.actor_loss + self.critic_loss + self.entropy_coef * entropy_mean
                # compute total loss
                self.optimizer.zero_grad()
                self.tot_losses_epoch.append(self.tot_loss.item())
                self.tot_loss.backward()
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)   
                self.optimizer.step()
        self.update_summary()
    