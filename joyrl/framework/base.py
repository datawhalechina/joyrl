#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 17:30:36
LastEditor: JiangJi
LastEditTime: 2023-12-02 21:59:10
Discription: 
'''
import ray
from joyrl.framework.config import MergedConfig
from joyrl.framework.message import Msg

class Moduler(object):
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        self.cfg = cfg
        if kwargs.get('use_ray', None) is not None:
            self.use_ray = kwargs['use_ray']
        else:
            self.use_ray = ray.is_initialized() 
    def _t_start(self):
        ''' start threads
        '''
        raise NotImplementedError
    
    def _p_start(self):
        ''' start processes
        '''
        raise NotImplementedError
    
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        raise NotImplementedError

    def init(self):
        ''' init module
        '''
        raise NotImplementedError
    
    def run(self):
        ''' run module
        '''
        raise NotImplementedError

    def ray_run(self):
        ''' asyn run module in ray
        '''
        raise NotImplementedError

