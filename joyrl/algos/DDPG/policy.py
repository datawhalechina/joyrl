#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-07-28 11:08:50
Discription: 
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.noise import MultiHeadActionNoise
from joyrl.algos.base.network import *
from .model import Model

class Policy(BasePolicy):
    def __init__(self,cfg, **kwargs) -> None:
        super(Policy, self).__init__(cfg, **kwargs)
        self.action_noise = MultiHeadActionNoise('ou',self.action_size_list)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.sample_count = 0 # sample count
        self.action_lows = [self.cfg.action_space_info.size[i][0] for i in range(len(self.action_size_list))]
        self.action_highs = [self.cfg.action_space_info.size[i][1] for i in range(len(self.action_size_list))]
        self.action_scales = [self.action_highs[i] - self.action_lows[i] for i in range(len(self.action_size_list))]
        self.action_biases = [self.action_highs[i] + self.action_lows[i] for i in range(len(self.action_size_list))]

    def create_model(self):
        ''' create graph and optimizer
        '''
        self.model = Model(self.cfg)
        self.target_model = Model(self.cfg)
        self.target_model.load_state_dict(self.model.state_dict()) # or use this to copy parameters

    def create_optimizer(self):
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.cfg.critic_lr)

    def create_summary(self):
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
            },
        }

    def update_summary(self):
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        self.summary['scalar']['policy_loss'] = self.policy_loss.item()
        self.summary['scalar']['value_loss'] = self.value_loss.item()

    def sample_action(self, state,  **kwargs):
        ''' sample action
        '''
        self.sample_count += 1
        state = self.process_sample_state(state)
        actor_outputs = self.model.actor(state)
        self.mu = torch.cat([actor_outputs[i]['mu'] for i in range(len(self.action_size_list))], dim=1)
        actions = get_model_actions(self.model, mode = 'sample', actor_outputs = actor_outputs)
        actions = self.action_noise.get_action(actions, t = self.sample_count) # add noise to action
        return actions
    
    def update_policy_transition(self):
        self.policy_transition = {'mu': self.mu.detach().cpu().numpy()}

    @torch.no_grad()
    def predict_action(self, state, **kwargs):
        ''' predict action
        '''
        state = self.process_sample_state(state)
        actor_outputs = self.model.actor(state)
        actions = get_model_actions(self.model, mode = 'predict', actor_outputs = actor_outputs)
        return actions
    
    def learn(self, **kwargs):
        ''' learn policy
        '''
        super().learn(**kwargs)
        actor_outputs = self.model.actor(self.states)
        mus = torch.cat([actor_outputs[i]['mu'] for i in range(len(self.action_size_list))], dim=1)
        self.policy_loss = - self.model.critic(self.states + [mus]).mean() * self.cfg.policy_loss_weight
        # calculate value loss
        tagert_model_actor_outputs = self.target_model.actor(self.next_states)
        next_mus = torch.cat([tagert_model_actor_outputs[i]['mu'] for i in range(len(self.action_size_list))], dim=1)
        # next_state_actions = torch.cat([next_states, next_actions], dim=1)
        target_values = self.target_model.critic(self.next_states + [next_mus])
        expected_values = self.rewards + self.gamma * target_values * (1.0 - self.dones)
        expected_values = torch.clamp(expected_values, self.cfg.value_min, self.cfg.value_max) # clip value
        actions = [ (self.actions[i] - self.action_biases[i])/ self.action_scales[i] for i in range(len(self.actions)) ]
        actions = torch.cat(actions, dim=1)
        actual_values = self.model.critic(self.states + [actions])
        self.value_loss = F.mse_loss(actual_values, expected_values.detach())
        self.tot_loss = self.policy_loss + self.value_loss
        # actor and critic update, the order is important
        self.actor_optimizer.zero_grad()
        self.policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        self.value_loss.backward()
        self.critic_optimizer.step()
        # soft update target network
        self.soft_update(self.model.actor, self.target_model.actor, self.tau)
        self.soft_update(self.model.critic, self.target_model.critic, self.tau)
        self.update_summary() # update summary
        
    def soft_update(self, curr_model, target_model, tau):
        ''' soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, curr_param in zip(target_model.parameters(), curr_model.parameters()):
            target_param.data.copy_(tau*curr_param.data + (1.0-tau)*target_param.data)
    