import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from joyrl.algos.base.action_layer import ActionLayerType
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import CriticNetwork, ActorNetwork
from joyrl.algos.base.noise import OUNoise

class Policy(BasePolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
        self.ou_noise = OUNoise(self.action_space)  
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.device = torch.device(cfg.device)
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary
        self.to(self.device)
        self.sample_count = 0 # sample count

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
        ''' create graph and optimizer
        '''
        critic_input_size_list = self.state_size_list + [[None, self.action_size_list[0]]]
        self.actor = ActorNetwork(self.cfg, input_size_list = self.state_size_list)
        self.critic = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.target_actor = ActorNetwork(self.cfg, input_size_list = self.state_size_list)
        self.target_critic = CriticNetwork(self.cfg, input_size_list = critic_input_size_list)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.create_optimizer() 

    def create_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
            },
        }
    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        self.summary['scalar']['policy_loss'] = self.policy_loss.item()
        self.summary['scalar']['value_loss'] = self.value_loss.item()

    def sample_action(self, state,  **kwargs):
        ''' sample action
        '''
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        _ = self.actor(state)
        action = self.actor.action_layers.get_actions()
        action = self.ou_noise.get_action(action, self.sample_count) # add noise to action
        return action[0]

    @torch.no_grad()
    def predict_action(self, state, **kwargs):
        ''' predict action
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        _ = self.actor(state)
        action = self.actor.action_layers.get_actions()
        return action[0]

    def learn(self, **kwargs):
        ''' train policy
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        # calculate policy loss
        self.policy_loss = - self.critic([states, self.actor(states)[0]['mu']]).mean() * self.cfg.policy_loss_weight
        # calculate value loss
        next_actions = self.target_actor(next_states)[0]['mu'].detach()
        # next_state_actions = torch.cat([next_states, next_actions], dim=1)
        target_values = self.target_critic([next_states, next_actions])
        expected_values = rewards + self.gamma * target_values * (1.0 - dones)
        expected_values = torch.clamp(expected_values, self.cfg.value_min, self.cfg.value_max) # clip value
        actual_values = self.critic([states, actions])
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
        self.soft_update(self.actor, self.target_actor, self.tau)
        self.soft_update(self.critic, self.target_critic, self.tau)
        self.update_summary() # update summary
        
    def soft_update(self, curr_model, target_model, tau):
        ''' soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, curr_param in zip(target_model.parameters(), curr_model.parameters()):
            target_param.data.copy_(tau*curr_param.data + (1.0-tau)*target_param.data)
    