#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-24 15:09:47
LastEditor: JiangJi
LastEditTime: 2024-01-13 18:26:43
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
        action = np.argmax(self.Q_table[str(state)])
        return action
    
    def learn(self, **kwargs):
        state, action, reward, next_state, done = kwargs.get('state'), kwargs.get('action'), kwargs.get('reward'), kwargs.get('next_state'), kwargs.get('done')
        Q_predict = self.Q_table[str(state)][action] 
        if done: 
            Q_target = reward 
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
        self.loss = (Q_target - Q_predict) ** 2
        self.update_summary() # update summary
        