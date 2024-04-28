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
        self.action_high = torch.FloatTensor(self.action_space.high).to(self.device)
        self.action_low = torch.FloatTensor(self.action_space.low).to(self.device)
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32)
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary
        self.to(self.device)

    def get_action_size(self):
        ''' get action size
        '''
        # action_size must be [action_dim_1, action_dim_2, ...]
        self.action_size_list = [self.action_space.shape[0]]
        self.action_type_list = ['dpg']
        self.action_high_list = [self.action_space.high[0]]
        self.action_low_list = [self.action_space.low[0]]
        setattr(self.cfg, 'action_size_list', self.action_size_list)
        setattr(self.cfg, 'action_type_list', self.action_type_list)
        setattr(self.cfg, 'action_high_list', self.action_high_list)
        setattr(self.cfg, 'action_low_list', self.action_low_list)
    
    def create_graph(self):
        critic_input_size_list = self.state_size_list + [[None, self.action_size_list[0]]]
        self.actor = ActorNetwork(self.cfg, input_size_list = self.state_size_list)
        self.critic_1 = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.critic_2 = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.target_actor = ActorNetwork(self.cfg, input_size_list = self.state_size_list)
        self.target_critic_1 = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.target_critic_2 = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
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
            action_noise = self.expl_noise * np.random.normal(0, self.action_scale.cpu().detach().numpy(), size=self.action_size_list[0])
            action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
            return action

    @torch.no_grad()
    def predict_action(self, state,  **kwargs):
        state = [torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)]
        _ = self.actor(state)
        action = self.actor.action_layers.get_actions()
        return action[0]

    def learn(self, **kwargs):
        # state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
   
        # update critic
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = self.target_actor(next_states)[0]['mu']
        # next_actions = ((next_actions + noise) * self.action_scale + self.action_bias).clamp(-self.action_scale+self.action_bias, self.action_scale+ self.action_bias)
        next_actions = (next_actions * self.action_scale + self.action_bias + noise).clamp(self.action_low, self.action_high)
        target_q1, target_q2 = self.target_critic_1([next_states, next_actions]).detach(), self.target_critic_2([next_states, next_actions]).detach()
        target_q = torch.min(target_q1, target_q2) # shape:[train_batch_size,n_actions]
        target_q = rewards + self.gamma * target_q * (1 - dones)
        current_q1, current_q2 = self.critic_1([states, actions]), self.critic_2([states, actions])
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
        self.soft_update(self.critic_1, self.target_critic_1, self.tau)
        self.soft_update(self.critic_2, self.target_critic_2, self.tau)
        # Delayed policy updates
        if self.sample_count % self.policy_freq == 0:
            # compute actor loss
            act_ = self.actor(states)[0]['mu'] * self.action_scale + self.action_bias
            actor_loss = -self.critic_1([states, act_]).mean()
            self.policy_loss = actor_loss
            self.tot_loss = self.policy_loss + self.value_loss1 + self.value_loss2
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.target_actor, self.tau)   
        self.update_step += 1
        self.update_summary()

    def soft_update(self, curr_model, target_model, tau):
        ''' soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, curr_param in zip(target_model.parameters(), curr_model.parameters()):
            target_param.data.copy_(tau*curr_param.data + (1.0-tau)*target_param.data)
