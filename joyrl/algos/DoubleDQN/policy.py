#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-06-14 22:49:54
Discription: 
'''
import torch
import torch.nn as nn
from joyrl.algos.DQN.policy import Policy as DQNPolicy
class Policy(DQNPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)

    def learn(self, **kwargs):
        ''' learn policy
        '''
        self.prepare_data_before_learn(**kwargs)
        actor_outputs = self.model(self.states)['actor_outputs']
        target_actor_outputs = self.target_model(self.next_states)['actor_outputs']
        tot_loss = 0
        self.summary_loss = []
        for i in range(len(self.action_size_list)):
            actual_q_value = actor_outputs[i]['q_value'].gather(1, self.actions[i].long())
            next_q_values = target_actor_outputs[i]['q_value']
            next_target_q_values_action = next_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
            expected_q_value = self.rewards + self.gamma * next_target_q_values_action * (1 - self.dones)
            loss_i = nn.MSELoss()(actual_q_value, expected_q_value)
            tot_loss += loss_i
            self.summary_loss.append(loss_i.item())
        self.optimizer.zero_grad() 
        tot_loss.backward()
        # clip to avoid gradient explosion
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target net every C steps
        if self.update_step % self.target_update == 0: 
            self.target_model.load_state_dict(self.model.state_dict())
        self.update_step += 1
        self.update_summary() # update summary
        
        