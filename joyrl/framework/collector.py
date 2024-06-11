#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-06-11 13:46:02
Discription: 
'''
import time
import threading
from queue import Queue, Empty, Full
from collections import deque
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.utils import exec_method

class Collector(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.data_handler = kwargs['data_handler']
        self.training_data_que = kwargs['training_data_que']
        self.sample_data_que = kwargs['sample_data_que']
        self._get_training_data_time = time.time()
        self._consumed_exp_len = 0
        self._handle_exps_time = time.time()
        self._produced_exp_len = 0
        self._t_interval = 3
        self._t_start()

    def _t_start(self):
        exec_method(self.logger, 'info', 'get', "[Collector._t_start] Start collector!") 

    def run(self):
        sample_data_len = 0
        while True:
            exps = exec_method(self.sample_data_que, 'pop', 'get')
            if exps:
                sample_data_len += len(exps)
                self._handle_exps(exps)
            if sample_data_len >= self.cfg.batch_size // 2:
                training_data = self._get_training_data()
                if training_data:
                    exec_method(self.training_data_que, 'put', 'remote', training_data)
                    sample_data_len = 0
            
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            self._handle_exps(exps)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            return self._get_training_data()
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError

    def _handle_exps(self, exps):
        ''' handle exps
        '''
        exec_method(self.data_handler, 'add_exps', 'remote', exps)
        self._produced_exp_len += len(exps)
        if time.time() - self._handle_exps_time >= self._t_interval:
            exec_method(self.logger, 'info', 'remote', f"[Collector._handle_exps] SAMPLE PRODUCTION SPEED per second: {self._produced_exp_len/self._t_interval:.2f}")
            self._handle_exps_time = time.time()
            self._produced_exp_len = 0

    def _get_training_data(self):
        ''' get training data
        '''
        get_training_data_time = time.time()
        training_data = None
        while True:
            training_data = exec_method(self.data_handler, 'get_training_data', 'get')
            if training_data:
                self._consumed_exp_len += training_data['dones'].shape[0]
                break
            if self.cfg.is_learner_async:
                break
            if time.time() - get_training_data_time >= 0.05:
                # exec_method(self.logger, 'warning', 'remote', "[Collector._get_training_data] get training data timeout!")
                get_training_data_time = time.time()
                break
        if time.time() - self._get_training_data_time >= self._t_interval:
            exec_method(self.logger, 'info', 'remote', f"[Collector.pub_msg] SAMPLE CONSUMPTION SPEED per second: {self._consumed_exp_len/self._t_interval:.2f}")
            self._get_training_data_time = time.time()
            self._consumed_exp_len = 0
        return training_data
    