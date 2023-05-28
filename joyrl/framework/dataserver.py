#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-27 20:46:36
LastEditor: JiangJi
LastEditTime: 2023-05-27 20:46:40
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full

@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0 # current episode
        self.sample_count = 0 # sample count
        self.update_step = 0 # update step
        self.max_episode = cfg.max_episode
    def increase_episode(self):
        ''' increase episode
        '''
        self.curr_episode += 1
    def check_episode_limit(self):
        ''' check if episode reaches the max episode
        '''
        return self.curr_episode > self.max_episode
    def get_episode(self):
        ''' get current episode
        '''
        return self.curr_episode
    def increase_sample_count(self):
        ''' increase sample count
        '''
        self.sample_count += 1
    def get_sample_count(self):
        ''' get sample count
        '''
        return self.sample_count
    def increase_update_step(self):
        ''' increase update step
        '''
        self.update_step += 1
    def get_update_step(self):
        ''' get update step
        '''
        return self.update_step

