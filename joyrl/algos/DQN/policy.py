import torch
import torch.nn as nn
import math,random
import numpy as np
from joyrl.algos.base.policy import BasePolicy
from joyrl.algos.base.network import QNetwork

class Policy(BasePolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
        self.gamma = cfg.gamma  
        # e-greedy parameters
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.target_update = cfg.target_update
        self.sample_count = 0
        self.update_step = 0
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary
        self.to(self.device)
        
    def create_graph(self):

        self.policy_net = QNetwork(self.cfg,self.state_size_list).to(self.device)
        self.target_net = QNetwork(self.cfg,self.state_size_list).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.create_optimizer()

    def sample_action(self, state,  **kwargs):
        ''' sample action
        '''
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            # before update, the network inference time may be longer
            action = self.predict_action(state) 
        else:
            action = [self.action_space.sample()]
        return action
    
    @torch.no_grad()
    def predict_action(self,state, **kwargs):
        ''' predict action
        '''
        state = [torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)]
        _ = self.policy_net(state)
        actions = self.policy_net.action_layers.get_actions()
        return actions
    
    def learn(self, **kwargs):
        ''' learn policy
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        # compute current Q values
        _ = self.policy_net(states)
        q_values = self.policy_net.action_layers.get_qvalues()
        actual_qvalues = q_values.gather(1, actions)
        # compute next max q value
        _ = self.target_net(next_states)
        next_q_values_max = self.target_net.action_layers.get_qvalues().max(1)[0].unsqueeze(dim=1)
        # compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_max
        # compute loss
        self.loss = nn.MSELoss()(actual_qvalues, target_q_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target net every C steps
        if self.update_step % self.target_update == 0: 
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_step += 1
        self.update_summary() # update summary
 