#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-06-14 09:34:00
Discription: 
'''
import time
import copy
import threading
from ray.util.queue import Queue as RayQueue
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.utils import exec_method, load_model_meta, save_model_meta

class PolicyMgr(Moduler):
    ''' model manager
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.policy = copy.deepcopy(kwargs['policy'])
        self._latest_model_params_dict = kwargs.get('latest_model_params_dict', None)
        self._saved_model_que = RayQueue(maxsize = 64)
        self._model_meta_que = RayQueue(maxsize = 64)
        self._t_start()
        
    def _t_start(self):
        exec_method(self.logger, 'info', 'get', "[PolicyMgr._t_start] Start policy manager!")
        self._t_save_policy = threading.Thread(target=self._save_policy)
        self._t_save_policy.setDaemon(True)
        self._t_save_policy.start()
        self._t_save_model_meta = threading.Thread(target=self._save_model_meta)
        self._t_save_model_meta.setDaemon(True)
        self._t_save_model_meta.start()
    
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.POLICY_MGR_PUT_MODEL_PARAMS:
            self._put_model_params(msg_data)
        elif msg_type == MsgType.POLICY_MGR_PUT_MODEL_META: # not used in this version
            self._put_model_meta(msg_data)
        else:
            raise NotImplementedError
        
    def _put_model_params(self, msg_data):
        ''' put model params
        '''
        update_step, model_params = msg_data
        exec_method(self._latest_model_params_dict, 'set_value', 'get', {'step': update_step, 'model_params': model_params})
        if update_step % self.cfg.model_save_fre == 0:
            while True:
                try: # if queue is full, wait for 0.01s
                    self._saved_model_que.put((update_step, model_params), block=False)
                    break
                except:
                    exec_method(self.logger, 'warning', 'get', "[PolicyMgr._put_model_params] saved_model_que is full!")
                    # time.sleep(0.001)

    def _put_model_meta(self, msg_data):
        ''' put model meta
        ''' 
        model_meta = msg_data
        try: # if queue is full, wait for 0.01s
            self._model_meta_que.put(model_meta, block=False)
        except:
            exec_method(self.logger, 'warning', 'get', "[PolicyMgr._put_model_meta] model_meta_que is full!")        

    def _save_policy(self):
        ''' async run
        '''
        while True:
            while not self._saved_model_que.empty():
                update_step, model_params = self._saved_model_que.get()
                self.policy.put_model_params(model_params)
                self.policy.save_model(f"{self.cfg.model_dir}/{update_step}")
            time.sleep(0.01)

    def _save_model_meta(self):
        ''' save model meta
        '''
        while True:
            while not self._model_meta_que.empty():
                model_meta_type, model_meta = self._model_meta_que.get()
                old_model_meta = load_model_meta(f"{self.cfg.model_dir}")
                if old_model_meta.get(model_meta_type, None) is None:
                    old_model_meta[model_meta_type] = {}
                old_model_meta[model_meta_type].update(model_meta)
                save_model_meta(f"{self.cfg.model_dir}", old_model_meta)
            time.sleep(0.002)
