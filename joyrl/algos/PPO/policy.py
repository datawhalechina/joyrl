#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-26 13:21:39
Discription: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
from joyrl.algos.base.network import ActorCriticNetwork, CriticNetwork, ActorNetwork
from joyrl.algos.base.policy import BasePolicy

class Policy(BasePolicy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.independ_actor = cfg.independ_actor
        self.share_optimizer = cfg.share_optimizer
        self.ppo_type = 'clip' # clip or kl
        if self.ppo_type == 'kl':
            self.kl_target = cfg.kl_target 
            self.kl_lambda = cfg.kl_lambda 
            self.kl_beta = cfg.kl_beta
            self.kl_alpha = cfg.kl_alpha
        self.gamma = cfg.gamma
        # self.action_type_list = cfg.action_type
        # if self.action_type_list.lower() == 'continuous': # continuous action space
        #     self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        #     self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.critic_loss_coef = cfg.critic_loss_coef
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sgd_batch_size = cfg.sgd_batch_size
        self.create_graph()
        self.create_optimizer()
        self.create_summary()
        self.to(self.device)
    
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

    def create_graph(self):
        if not self.independ_actor:
            self.policy_net = ActorCriticNetwork(self.cfg, self.state_size_list).to(self.device)
        else:
            self.actor = ActorNetwork(self.cfg, self.state_size_list).to(self.device)
            self.critic = CriticNetwork(self.cfg, self.state_size_list).to(self.device)

    def create_optimizer(self):
        if self.share_optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr) 
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)  

    def update_policy_transition(self):
        self.policy_transition = {'value': self.value, 'log_prob': self.log_prob}

    def sample_action(self, state, **kwargs):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        if not self.independ_actor:
            raise NotImplementedError
        else:
            self.value = self.critic(state)
            _ = self.actor(state) # list
            actions = self.actor.action_layers.get_actions(mode = 'sample')
            self.log_prob = self.actor.action_layers.get_log_probs().detach().cpu().numpy().item()
        self.update_policy_transition()
        return actions
        # if self.action_type_list.lower() == 'continuous':
        #     mean = self.mu * self.action_scale + self.action_bias
        #     std = self.sigma
        #     dist = Normal(mean,std)
        #     action = dist.sample()
        #     action = torch.clamp(action, torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32), torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32))
        #     self.log_probs = dist.log_prob(action).detach()
        #     return action.detach().cpu().numpy()[0]
        # else:
        #     dist = Categorical(self.probs)
        #     action = dist.sample()
        #     self.log_probs = dist.log_prob(action).detach()
        #     return action.detach().cpu().numpy().item()
        # dist = Categorical(self.probs)
        # action = dist.sample()
        # self.log_probs = dist.log_prob(action).detach()
        # return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self, state, **kwargs):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        _ = self.actor(state) # list
        actions = self.actor.action_layers.get_actions(mode = 'predict')
        return actions
    
    def learn(self, **kwargs): 
        states, actions, next_states, rewards, dones, returns = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones'), kwargs.get('returns')
        old_log_probs  = kwargs.get('log_probs')
        torch_dataset = Data.TensorDataset(*states, actions, old_log_probs, returns)
        train_loader = Data.DataLoader(dataset = torch_dataset, batch_size = self.sgd_batch_size, shuffle = False, drop_last = False)
        for _ in range(self.k_epochs):
            for data in train_loader:
                old_states = []
                for i in range(len(states)):
                    old_states.append(data[i])
                idx = len(states)
                old_actions = data[idx]
                old_log_probs = data[idx+1]
                returns = data[idx+2]
                values = self.critic(old_states.copy()) # detach to avoid backprop through the critic
                advantages = returns - values.detach() # shape:[batch_size,1]
                # get action probabilities
                _ = self.actor(old_states.copy()) # list
                old_actions = old_actions.squeeze(dim=1)
                new_log_probs = self.actor.action_layers.get_log_probs_action(old_actions).unsqueeze(dim=1)
                entropy_mean = self.actor.action_layers.get_mean_entropy()
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_log_probs - old_log_probs) # shape: [batch_size, 1]
                # compute surrogate loss
                surr1 = ratio * advantages # shape: [batch_size, 1]
                # print(surr1.shape, advantages.shape, ratio.shape)
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                self.actor_loss = - (torch.mean(torch.min(surr1, surr2)) + self.entropy_coef * entropy_mean)
                # compute critic loss
                self.critic_loss = self.cfg.critic_loss_coef * nn.MSELoss()(returns, values) # shape: [batch_size, 1]
                # compute total loss
                if self.share_optimizer:
                    self.optimizer.zero_grad()
                    self.tot_loss = self.actor_loss + self.critic_loss
                    self.tot_loss.backward()
                else:
                    self.actor_optimizer.zero_grad()
                    self.actor_loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()  
                    self.critic_loss.backward()
                    self.critic_optimizer.step()
            self.update_summary()


            