import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.noise import MultiHeadActionNoise
from joyrl.algos.base.network import *
from .model import Model

class Policy(BasePolicy):
    def __init__(self, cfg):
        super(Policy, self).__init__(cfg)
        self.gamma = cfg.gamma
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.policy_noise = cfg.policy_noise # noise added to target policy during critic update
        self.noise_clip = cfg.noise_clip # range to clip target policy noise
        self.action_noise = MultiHeadActionNoise('random',self.action_size_list, theta = cfg.expl_noise)
        self.action_lows = [self.cfg.action_space_info.size[i][0] for i in range(len(self.action_size_list))]
        self.action_highs = [self.cfg.action_space_info.size[i][1] for i in range(len(self.action_size_list))]
        self.action_scales = [self.action_highs[i] - self.action_lows[i] for i in range(len(self.action_size_list))]
        self.action_biases = [self.action_highs[i] + self.action_lows[i] for i in range(len(self.action_size_list))]
        self.policy_freq = cfg.policy_freq # policy update frequency
        self.tau = cfg.tau
        self.sample_count = 0
        self.update_step = 0
        self.explore_steps = cfg.explore_steps # exploration steps before training
    
    def create_model(self):
        ''' create graph and optimizer
        '''
        self.model = Model(self.cfg)
        self.target_model = Model(self.cfg)
        self.target_model.load_state_dict(self.model.state_dict()) # or use this to copy parameters

    def create_optimizer(self):
        self.actor_optimizer = optim.Adam(self.model.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.model.critic_1.parameters(), lr=self.cfg.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.model.critic_2.parameters(), lr=self.cfg.critic_lr)

    def create_summary(self):
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss1': 0.0,
                'value_loss2': 0.0,
            },
        }

    def update_summary(self):
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        try:
            self.summary['scalar']['policy_loss'] = self.policy_loss.item()
        except Exception as e:
            pass
        self.summary['scalar']['value_loss1'] = self.value_loss1.item()
        self.summary['scalar']['value_loss2'] = self.value_loss2.item()

    def sample_action(self, state,  **kwargs):
        self.sample_count += 1
        if self.sample_count < self.explore_steps:
            return get_model_actions(self.model, mode = 'random', actor_outputs = [{}] * len(self.action_size_list))
        else:
            actions = self.predict_action(state, **kwargs)
            actions = self.action_noise.get_action(actions, t = self.sample_count) # add noise to action
            return actions

    @torch.no_grad()
    def predict_action(self, state,  **kwargs):
        state = self.process_sample_state(state)
        actor_outputs = self.model.actor(state)
        actions = get_model_actions(self.model, mode = 'predict', actor_outputs = actor_outputs)
        return actions

    def learn(self, **kwargs):
        super().learn(**kwargs)
        # update critic
        next_actor_outputs = self.target_model.actor(self.next_states)
        # next_actions = get_model_actions(self.target_model, mode = 'predict', actor_outputs = actor_outputs)
        next_mus = torch.cat([next_actor_outputs[i]['mu'] for i in range(len(self.action_size_list))], dim=1)
        # noise = (torch.randn_like(next_mus) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # next_mus_noised = (next_mus + noise).clamp(self.action_low, self.action_high)
        target_q1, target_q2 = self.target_model.critic_1(self.next_states+ [next_mus]).detach(), self.target_model.critic_2(self.next_states+ [next_mus]).detach()
        target_q = torch.min(target_q1, target_q2) # shape:[train_batch_size,n_actions]
        target_q = self.rewards + self.gamma * target_q * (1 - self.dones)
        actions = [ (self.actions[i] - self.action_biases[i])/ self.action_scales[i] for i in range(len(self.actions)) ]
        actions = torch.cat(actions, dim=1)
        current_q1, current_q2 = self.model.critic_1(self.states + [actions]), self.model.critic_2(self.states + [actions])
        # compute critic loss
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.value_loss1, self.value_loss2 = critic_1_loss, critic_2_loss
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        self.soft_update(self.model.critic_1, self.target_model.critic_1, self.tau)
        self.soft_update(self.model.critic_2, self.target_model.critic_2, self.tau)
        # Delayed policy updates
        if self.update_step % self.policy_freq == 0:
            # compute actor loss
            actor_outputs = self.model.actor(self.states)
            mus = torch.cat([actor_outputs[i]['mu'] for i in range(len(self.action_size_list))], dim=1)
            actor_loss = -self.model.critic_1(self.states + [mus]).mean()
            self.policy_loss = actor_loss
            self.tot_loss = self.policy_loss + self.value_loss1 + self.value_loss2
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.model.actor, self.target_model.actor, self.tau)
        self.update_step += 1
        self.update_summary()

    def soft_update(self, curr_model, target_model, tau):
        ''' soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, curr_param in zip(target_model.parameters(), curr_model.parameters()):
            target_param.data.copy_(tau*curr_param.data + (1.0-tau)*target_param.data)
