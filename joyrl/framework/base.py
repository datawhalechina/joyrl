#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 17:30:36
LastEditor: JiangJi
LastEditTime: 2024-01-04 23:48:51
Discription: 
'''
import ray
from joyrl.framework.config import MergedConfig
from joyrl.framework.message import Msg
from joyrl.utils.utils import Logger, create_module

class Moduler(object):
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        self.cfg = cfg
        self.use_ray = kwargs.get('use_ray', False)
        # self.logger = Logger(self.cfg.log_dir)
        self.logger = create_module(Logger, self.use_ray, {'num_cpus':0} , self.cfg.log_dir)
        
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

