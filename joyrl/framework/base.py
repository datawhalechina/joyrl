#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 17:30:36
LastEditor: JiangJi
LastEditTime: 2024-06-02 10:50:42
Discription: 
'''
import ray
from joyrl.framework.config import MergedConfig
from joyrl.framework.message import Msg
from joyrl.framework.utils import Logger, create_module


class Moduler(object):
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        self.cfg = cfg
        self.name = kwargs.get('name', 'Moduler')
        self.logger = Logger(self.cfg.log_dir, log_name = self.name)
        
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


