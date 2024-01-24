#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-21 17:45:27
Discription: 
'''
import time
import copy
import threading
import torch
from typing import Dict, List
from queue import Queue
from ray.util.queue import Queue as RayQueue
from joyrl.framework.message import Msg, MsgType
from joyrl.algos.base.policy import BasePolicy
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.utils.utils import exec_method

class PolicyMgr(Moduler):
    ''' model manager
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.policy = copy.deepcopy(kwargs['policy'])
        self._latest_model_params_dict = {'step': 0, 'model_params': self.policy.get_model_params()}
        self._saved_model_que = RayQueue(maxsize = 128) if self.use_ray else Queue(maxsize = 128)
        self._t_start()
        
    def _t_start(self):
        exec_method(self.logger, 'info', False, "[PolicyMgr._t_start] Start policy manager!")
        self._t_save_policy = threading.Thread(target=self._save_policy)
        self._t_save_policy.setDaemon(True)
        self._t_save_policy.start()
    
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.MODEL_MGR_PUT_MODEL_PARAMS:
            self._put_model_params(msg_data)
        elif msg_type == MsgType.MODEL_MGR_GET_MODEL_PARAMS:
            return self._get_model_params()
        else:
            raise NotImplementedError
        
    def _put_model_params(self, msg_data):
        ''' put model params
        '''
        update_step, model_params = msg_data
        if update_step > self._latest_model_params_dict['step']:
            self._latest_model_params_dict['step'] = update_step
            self._latest_model_params_dict['model_params'] = model_params
        if update_step % self.cfg.model_save_fre == 0:
            while True:
                try: # if queue is full, wait for 0.01s
                    self._saved_model_que.put((update_step, model_params), block=False)
                    break
                except:
                    exec_method(self.logger, 'warning', True, "[PolicyMgr._put_model_params] saved_model_que is full!")
                    # time.sleep(0.001)

    def _get_model_params(self):
        ''' get policy
        '''
        return self._latest_model_params_dict['model_params']

    def _save_policy(self):
        ''' async run
        '''
        while True:
            while not self._saved_model_que.empty():
                update_step, model_params = self._saved_model_que.get()
                self.policy.put_model_params(model_params)
                self.policy.save_model(f"{self.cfg.model_dir}/{update_step}")
            time.sleep(0.02)
    

