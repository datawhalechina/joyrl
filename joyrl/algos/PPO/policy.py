#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-23 22:08:15
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
            self.policy_net = ActorCriticNetwork(self.cfg, self.state_size, self.action_size_list, self.action_type_list)
        else:
            self.actor = ActorNetwork(self.cfg, self.state_size, self.action_size_list, self.action_type_list)
            self.critic = CriticNetwork(self.cfg, self.state_size, self.action_size_list)

    def create_optimizer(self):
        if self.share_optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr) 
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
    @torch.no_grad()         
    def get_action(self, state, mode='sample', **kwargs):
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        if not self.independ_actor:
            if self.action_type_list.lower() == 'continuous':
                self.value, self.mu, self.sigma = self.policy_net(state)
            else:
                self.probs = self.policy_net(state)
        else:
            self.value = self.critic(state)
            output = self.actor(state)
            self.probs = output['probs']
            # if self.action_type_list.lower() == 'continuous':
            #     self.mu, self.sigma = self.actor(state)
            # else:
            #     output = self.actor(state)
            #     self.probs = output['probs']
        if self.cfg.mode == 'train':
            action = self.sample_action(**kwargs)
            self.update_policy_transition()
        elif self.cfg.mode  == 'test':
            action = self.predict_action(**kwargs)
        else:
            raise NameError('mode must be sample or predict')
        return action
    def update_policy_transition(self):
        self.policy_transition = {'value': self.value, 'probs': self.probs, 'log_probs': self.log_probs}
        # if self.action_type_list.lower() == 'continuous':
        #     self.policy_transition = {'value': self.value, 'mu': self.mu, 'sigma': self.sigma}
        # else:
        #     self.policy_transition = {'value': self.value, 'probs': self.probs, 'log_probs': self.log_probs}
    def sample_action(self, state, **kwargs):
        if not self.independ_actor:
            pass
        else:
            self.value = self.critic(state)
            action_output = self.actor(state) # list
            self.probs = action_output['probs']
            action = self.actor.action_layers.get_actions(self.probs, **kwargs)

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
        dist = Categorical(self.probs)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()
    def predict_action(self, **kwargs):
        return torch.argmax(self.probs).detach().cpu().numpy().item()
        if self.action_type_list.lower() == 'continuous':
            return self.mu.detach().cpu().numpy()[0]
        else:
            return torch.argmax(self.probs).detach().cpu().numpy().item()
    def learn(self, **kwargs): 
        states, actions, next_states, rewards, dones, returns = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones'), kwargs.get('returns')
        # if self.action_type_list.lower() == 'continuous':      
        #     mus, sigmas = kwargs.get('mu'), kwargs.get('sigma')
        #     mus = torch.stack(mus, dim=0).to(device=self.device, dtype=torch.float32)
        #     sigmas = torch.stack(sigmas, dim=0).to(device=self.device, dtype=torch.float32)
        #     means = mus * self.action_scale + self.action_bias
        #     stds = sigmas
        #     dists = Normal(means,stds)
        #     old_log_probs = dists.log_prob(torch.tensor(np.array(actions), device=self.device, dtype=torch.float32)).detach()
        #     old_probs = torch.exp(old_log_probs)
        # else:
        #     old_probs, old_log_probs  = kwargs.get('probs'), kwargs.get('log_probs')
        #     old_probs = torch.stack(old_probs, dim=0).to(device=self.device, dtype=torch.float32)  # shape:[batch_size,n_actions]
        #     old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[batch_size,1]
        old_probs, old_log_probs  = kwargs.get('probs'), kwargs.get('log_probs')
        old_probs = torch.stack(old_probs, dim=0).to(device=self.device, dtype=torch.float32)  # shape:[batch_size,n_actions]
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[batch_size,1]
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        # actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[batch_size,1]
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32).unsqueeze(1) # shape:[batch_size,1]
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        # returns = self._compute_returns(rewards, dones) # shape:[batch_size,1]  
        returns = torch.tensor(returns.copy(), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        torch_dataset = Data.TensorDataset(states, actions, old_probs, old_log_probs,returns)
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.sgd_batch_size, shuffle=False,drop_last=False)
        for _ in range(self.k_epochs):
            for _, (old_states, old_actions, old_probs, old_log_probs, returns) in enumerate(train_loader):
                # compute advantages
                values = self.critic(old_states) # detach to avoid backprop through the critic
                advantages = returns - values.detach() # shape:[batch_size,1]
                # get action probabilities
                output = self.actor(old_states) 
                new_probs = output['probs'] # shape:[batch_size,n_actions]
                dist = Categorical(new_probs)
                # get new action probabilities
                new_log_probs = dist.log_prob(old_actions.squeeze(dim=1)).unsqueeze(dim=1) # shape:[batch_size,1]
                # if self.action_type_list.lower() == 'continuous':
                #     mu, sigma = self.actor(old_states)
                #     mean = mu * self.action_scale + self.action_bias
                #     std = sigma
                #     dist = Normal(mean, std)
                #     new_log_probs = dist.log_prob(old_actions)
                # else:
                #     output = self.actor(old_states) 
                #     new_probs = output['probs'] # shape:[batch_size,n_actions]
                #     dist = Categorical(new_probs)
                #     # get new action probabilities
                #     new_log_probs = dist.log_prob(old_actions.squeeze(dim=1)).unsqueeze(dim=1) # shape:[batch_size,1]
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_log_probs - old_log_probs) # shape: [batch_size, 1]
                # compute surrogate loss
                surr1 = ratio * advantages # shape: [batch_size, 1]
                if self.ppo_type == 'clip':
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                    self.actor_loss = - (torch.mean(torch.min(surr1, surr2)) + torch.mean(self.entropy_coef * dist.entropy()))
                elif self.ppo_type == 'kl':
                    kl_mean = F.kl_div(torch.log(new_probs.detach()), old_probs.unsqueeze(1),reduction='mean') # KL(input|target),new_probs.shape: [batch_size, n_actions]
                    # kl_div = torch.mean(new_probs * (torch.log(new_probs) - torch.log(old_probs)), dim=1) # KL(new|old),new_probs.shape: [batch_size, n_actions]
                    surr2 = self.kl_lambda * kl_mean
                    # surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                    self.actor_loss = - (surr1.mean() + surr2 + self.entropy_coef * dist.entropy().mean())
                    if kl_mean > self.kl_beta * self.kl_target:
                        self.kl_lambda *= self.kl_alpha
                    elif kl_mean < 1/self.kl_beta * self.kl_target:
                        self.kl_lambda /= self.kl_alpha
                else:
                    raise NameError("ppo_type must be 'clip' or 'kl'")
                # compute critic loss
                
                self.critic_loss = nn.MSELoss()(returns, values) # shape: [batch_size, 1]
                # compute total loss
                if self.share_optimizer:
                    self.optimizer.zero_grad()
                    self.tot_loss = self.actor_loss + self.critic_loss_coef* self.critic_loss
                    self.tot_loss.backward()
                else:
                    self.actor_optimizer.zero_grad()
                    self.actor_loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.zero_grad()  
                    self.critic_loss.backward()
                    self.critic_optimizer.step()
        self.update_summary()


        