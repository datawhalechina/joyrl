import numpy as np
import torch
import torch.nn.functional as F
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import CriticNetwork, ActorNetwork


class Policy(BasePolicy):
    def __init__(self, cfg):
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.actor_lr = cfg.actor_lr
        self.critic_lr = cfg.critic_lr
        self.policy_noise = cfg.policy_noise # noise added to target policy during critic update
        self.noise_clip = cfg.noise_clip # range to clip target policy noise
        self.expl_noise = cfg.expl_noise # std of Gaussian exploration noise
        self.policy_freq = cfg.policy_freq # policy update frequency
        self.tau = cfg.tau
        self.sample_count = 0
        self.update_step = 0
        self.explore_steps = cfg.explore_steps # exploration steps before training
        self.device = torch.device(cfg.device)
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary
    
    def create_graph(self):
        self.state_size, self.action_size = self.get_state_action_size()
        self.n_actions = self.action_size[-1]
        self.input_head_size = [None, self.state_size[-1]+self.action_size[-1]]
        # Actor
        self.actor = ActorNetwork(self.cfg, self.state_size, self.action_space).to(self.device)
        self.actor_target = ActorNetwork(self.cfg, self.state_size, self.action_space).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # critice - 2Q
        self.critic_1 = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
        self.critic_2 = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
        self.critic_1_target = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
        self.critic_2_target = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.create_optimizer() 

    def create_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = self.critic_lr)

    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss1': 0.0,
                'value_loss2': 0.0,
            },
        }

    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
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
            return self.action_space.sample()
        else:
            action = self.predict_action(state, **kwargs)
            action_noise = np.random.normal(0, self.action_scale.cpu().numpy()[0] * self.expl_noise, size=self.n_actions)
            action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
            return action

    @torch.no_grad()
    def predict_action(self, state,  **kwargs):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action = self.actor(state)
        action = self.action_scale * action + self.action_bias
        return action.detach().cpu().numpy()[0]

    def learn(self, **kwargs):
        # state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, next_state, reward, done = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        # convert to tensor
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)
        # update critic
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # print ("")
        next_action = ((self.actor_target(next_state) + noise) * self.action_scale + self.action_bias).clamp(-self.action_scale+self.action_bias, self.action_scale+self.action_bias)
        next_sa = torch.cat([next_state, next_action], 1) # shape:[train_batch_size,n_states+n_actions]
        target_q1, target_q2 = self.critic_1_target(next_sa).detach(), self.critic_2_target(next_sa).detach()
        target_q = torch.min(target_q1, target_q2) # shape:[train_batch_size,n_actions]
        target_q = reward + self.gamma * target_q * (1 - done)
        sa = torch.cat([state, action], 1)
        current_q1, current_q2 = self.critic_1(sa), self.critic_2(sa)
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
        # Delayed policy updates
        if self.sample_count % self.policy_freq == 0:
            # compute actor loss
            actor_loss = -self.critic_1(torch.cat([state, self.actor(state)], 1)).mean()
            self.policy_loss = actor_loss
            self.tot_loss = self.policy_loss + self.value_loss1 + self.value_loss2
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic_1, self.critic_1_target, self.tau)
            self.soft_update(self.critic_2, self.critic_2_target, self.tau)
        self.update_step += 1
        self.update_summary()

    def soft_update(self, curr_model, target_model, tau):
        ''' soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, curr_param in zip(target_model.parameters(), curr_model.parameters()):
            target_param.data.copy_(tau*curr_param.data + (1.0-tau)*target_param.data)
