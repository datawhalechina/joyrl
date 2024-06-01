#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-06-01 16:52:10
Discription: 
'''
import ray
import time
import multiprocessing as mp
from queue import Queue, Empty, Full
import threading
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.utils.utils import exec_method
from joyrl.framework.utils import DeQueue

class Collector(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.data_handler = kwargs['data_handler']
        self._get_training_data_time = time.time()
        self._consumed_exp_len = 0
        self._handle_exps_time = time.time()
        self._produced_exp_len = 0
        self._t_interval = 3
        self._t_start()

    def _t_start(self):
        exec_method(self.logger, 'info', 'get', "[Collector._t_start] Start collector!") 

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            self._handle_exps(exps)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            try:
                training_data = self._get_training_data()
                if training_data is None:
                    return None
                self._consumed_exp_len += training_data['dones'].shape[0]
                if time.time() - self._get_training_data_time >= self._t_interval:
                    exec_method(self.logger, 'info', 'remote', f"[Collector.pub_msg] SAMPLE CONSUMPTION SPEED per second: {self._consumed_exp_len/self._t_interval:.2f}")
                    self._get_training_data_time = time.time()
                    self._consumed_exp_len = 0
                return training_data
            except Empty:
                # exec_method(self.logger, 'warning', 'get', "[Collector.pub_msg] training_data_que is empty!")
                return None
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError

    def _handle_exps(self, exps):
        ''' handle exps
        '''
        self._produced_exp_len += len(exps)
        exec_method(self.data_handler, 'add_exps', 'remote', exps)
        if time.time() - self._handle_exps_time >= self._t_interval:
            exec_method(self.logger, 'info', 'remote', f"[Collector._handle_exps] SAMPLE PRODUCTION SPEED per second: {self._produced_exp_len/self._t_interval:.2f}")
            self._handle_exps_time = time.time()
            self._produced_exp_len = 0
            
    def _prepare_training_data(self):
        ''' 
        '''
        while True:
            training_data = self._get_training_data()
            if training_data is not None:
                self._training_data_que.append(training_data)
                # try:
                #     self._training_data_que.put(training_data, block = False, timeout=0.1)
                # except Full:
                #     pass
            # if not self._training_data_que.full():
            #     training_data = self._get_training_data()
            #     if training_data is not None:
            #         try:
            #             self._training_data_que.put(training_data, block = True, timeout=0.1)
            #             consumed_exp_len += training_data['dones'].shape[0]
            #             if time.time() - s_t >= t_interval:
            #                 exec_method(self.logger, 'info', 'remote', f"[Collector._prepare_training_data] SAMPLE CONSUMPTION SPEED per second: {consumed_exp_len/t_interval:.2f}")
            #                 s_t = time.time()
            #                 consumed_exp_len = 0
            #         except Full:
            #             pass
                        # exec_method(self.logger, 'warning', 'get', "[Collector._prepare_training_data] training_data_que is full!")
            # time.sleep(0.002)
            
    def _get_training_data(self):
        training_data = exec_method(self.data_handler, 'sample_training_data', 'get')
        return training_data
    
    def get_buffer_length(self):
        return len(self.data_handler.buffer)
