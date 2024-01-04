#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-04 13:47:44
Discription: 
'''
import ray
from ray.util.queue import Queue as RayQueue
import multiprocessing as mp
from queue import Queue, Empty, Full
import threading
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.algos.base.data_handler import BaseDataHandler
from joyrl.framework.base import Moduler
from joyrl.utils.utils import memory_profile

class Collector(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.data_handler = kwargs['data_handler']
        self._training_data_que = Queue(maxsize = 1)
        self._raw_exps_que = Queue(maxsize = 128) if not self.use_ray else RayQueue(maxsize = 128)
        self._t_start()

    def _t_start(self):
        self._t_handle_exps = threading.Thread(target=self._handle_exps)
        self._t_handle_exps.setDaemon(True)
        self._t_handle_exps.start()
        self._t_prepare_training_data = threading.Thread(target=self._prepare_training_data)
        self._t_prepare_training_data.setDaemon(True)
        self._t_prepare_training_data.start()
    

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            try:
                self._raw_exps_que.put(exps, block = False)
            except Full:
                self.logger.warning("[Collector.pub_msg] raw_exps_que is full!")
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            try:
                return self._training_data_que.get(block = False)
            except:
                return None
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError

    def _handle_exps(self):
        ''' handle exps
        '''
        while True:
            exps = self._raw_exps_que.get()
            self.data_handler.add_exps(exps) # add exps to data handler
            
    def _prepare_training_data(self):
        ''' 
        '''
        while True:
            training_data = self.data_handler.sample_training_data()
            if training_data is None:
                continue
            else:
                try:
                    self._training_data_que.put(training_data, block = False)
                except Full:
                    # self.logger.warning("[Collector._sample_training_data] training_data_que is full!")
                    pass

    def _get_training_data(self):
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    
    def handle_data_after_learn(self, policy_data_after_learn, *args, **kwargs):
        return 
    
    def get_buffer_length(self):
        return len(self.data_handler.buffer)
