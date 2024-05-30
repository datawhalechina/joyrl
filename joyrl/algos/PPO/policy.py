#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-05-28 00:17:20
Discription: 
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
from joyrl.algos.base.network import ActorCriticNetwork, CriticNetwork, ActorNetwork
from joyrl.algos.base.policy import BasePolicy
from joyrl.framework.config import MergedConfig

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, std=0.0):
        super(Model, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.log_std = nn.Parameter(torch.ones(1, output_dim) * std)
        
        # self.apply(init_linear_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        mu = torch.tanh(mu)
        std = self.log_std.exp().expand_as(mu)
        return mu, std, value

class Policy(BasePolicy):
    def __init__(self, cfg: MergedConfig) -> None:
        super(Policy, self).__init__(cfg)
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

        
        # self.model = Model(self.state_size_list[0][1], self.action_size_list[0], 256, std=0.0).to(self.device)

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.cfg.lr)
        # if not self.independ_actor:
        #     self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr) 
        # else:
        #     self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        #     self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)  

    def update_policy_transition(self):
        self.policy_transition = {'value': self.value.detach().cpu().numpy().item(), 'log_prob': self.log_prob}

    def sample_action(self, state, **kwargs):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        # single state shape must be [batch_size, state_dim]
        if state.dim() == 1: 
            state = state.unsqueeze(dim=0)
        self.value, self.actor_outputs = self.model(state)
        actions, self.log_prob = self.model.action_layers.get_actions_and_log_probs(mode = 'sample', actor_outputs = self.actor_outputs)
        self.update_policy_transition()
        return actions

    def compute_return_mc(self, rewards, masks):
        returns = []
        discounted_return = 0
        for step in reversed(range(len(rewards))):
            discounted_return = rewards[step] + self.gamma * discounted_return * masks[step]
            returns.insert(0, discounted_return)
        # return returns
        # print("return",returns)
        # print("masks",masks)
        # print(rewards)
        # normalize can help to stabilize training
        returns = torch.cat(returns)
        normed_returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        # normed_returns = []
        # for ret in returns:
        #     normed_returns.append((ret - ret.mean()) / (ret.std() + 1e-5))
        return normed_returns

    @torch.no_grad()
    def predict_action(self, state, **kwargs):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        # single state shape must be [batch_size, state_dim]
        if state.dim() == 1: 
            state = state.unsqueeze(dim=0)
        self.value, self.actor_outputs = self.model(state)
        actions = self.model.action_layers.get_actions(mode = 'predict', actor_outputs = self.actor_outputs)
        return actions

    def learn(self, **kwargs):
        states, actions, next_states, rewards, dones, returns = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones'), kwargs.get('returns')
        old_log_probs  = kwargs.get('log_probs')
        torch_dataset = Data.TensorDataset(*states, *actions, old_log_probs, returns)
        train_loader = Data.DataLoader(dataset = torch_dataset, batch_size = self.sgd_batch_size, shuffle = True)
        self.actor_losses_epoch, self.critic_losses_epoch, self.tot_losses_epoch = [], [], []
        for _ in range(self.k_epochs):
            for data in train_loader:
                # multi-head state
                old_states = []
                for i in range(len(states)):
                    old_states.append(data[i])
                idx = len(states)
                # multi-head action
                old_actions = []
                for i in range(len(actions)):
                    old_actions.append(data[idx+i])
                idx += len(actions)
                old_log_probs = data[idx]
                returns = data[idx+1]
                values, actor_outputs = self.model(old_states)
                new_log_probs = self.model.action_layers.get_log_probs_action(actor_outputs, old_actions)
                # new_log_probs = self.model.action_layers.get_log_probs_action(old_actions)
                entropy_mean = self.model.action_layers.get_mean_entropy(actor_outputs)
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
    