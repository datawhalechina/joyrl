#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-01-11 13:01:30
LastEditor: JiangJi
LastEditTime: 2024-01-11 13:04:51
Discription: 
'''
import math
import numpy as np
from collections import defaultdict
from joyrl.algos.base.policy import ToyPolicy

class Policy(ToyPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.lr = cfg.lr 
        self.gamma = cfg.gamma 
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.sample_count = 0
        self.next_action = None
        self.create_summary()

    
    def sample_action(self, state, **kwargs):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) #  select the action with max Q value
        else:
            action = np.random.choice(self.n_actions) # random select an action
        return action
    
    def predict_action(self, state, **kwargs):
        if self.next_action is None:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = self.next_action
        return action
    
    def learn(self, **kwargs):
        state, action, reward, next_state, done = kwargs.get('state'), kwargs.get('action'), kwargs.get('reward'), kwargs.get('next_state'), kwargs.get('done')
        Q_predict = self.Q_table[str(state)][action] 
        next_action = self.get_action(next_state) # next action
        self.next_action = next_action
        if done: 
            Q_target = reward 
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][self.next_action]
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
        self.loss = (Q_target - Q_predict) ** 2
        self.update_summary() # update summary
        