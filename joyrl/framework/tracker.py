#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-12-02 23:29:10
Discription: 
'''
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler

class Tracker(Moduler):
    ''' tacker global information
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.global_episode = 0 # current global episode
        self.global_sample_count = 0 # global sample count
        self.global_update_step = 0 # global update step
        self.max_episode = cfg.max_episode # max episode

    def pub_msg(self, msg: Msg):
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.TRACKER_GET_EPISODE:
            return self._get_episode()
        elif msg_type == MsgType.TRACKER_INCREASE_EPISODE:
            episode_delta = 1 if msg_data is None else msg_data
            self._increase_episode(i = episode_delta)
        elif msg_type == MsgType.TRACKER_GET_UPDATE_STEP:
            return self._get_update_step()
        elif msg_type == MsgType.TRACKER_INCREASE_UPDATE_STEP:
            update_step_delta = 1 if msg_data is None else msg_data
            self._increase_update_step(i = update_step_delta)
        elif msg_type == MsgType.TRACKER_CHECK_TASK_END:
            return self._check_task_end()
        else:
            raise NotImplementedError
        
    def _increase_episode(self, i: int = 1):
        ''' increase episode
        '''
        self.global_episode += i
        
    def _get_episode(self):
        ''' get current episode
        '''
        return self.global_episode
    
    def _check_task_end(self):
        ''' check if episode reaches the max episode
        '''
        if self.max_episode < 0:
            return False
        return self.global_episode >= self.max_episode
    
    def _increase_update_step(self, i: int = 1):
        ''' increase update step
        '''
        self.global_update_step += i
        
    def _get_update_step(self):
        ''' get update step
        '''
        return self.global_update_step