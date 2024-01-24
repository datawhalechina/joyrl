#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-15 23:49:03
Discription: 
'''
import ray
import copy
import time
from typing import Tuple
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.recorder import Recorder
from joyrl.utils.utils import exec_method, create_module

class Learner:
    ''' learner
    '''
    def __init__(self, cfg : MergedConfig, **kwargs) -> None:
        self.cfg = cfg
        self.id = kwargs.get('id', 0)
        self.policy = kwargs.get('policy', None)
        self.policy_mgr = kwargs.get('policy_mgr', None)
        self.collector = kwargs.get('collector', None)
        self.tracker = kwargs.get('tracker', None)
        self.recorder = kwargs.get('recorder', None)
        self.logger = kwargs['logger']
        self.use_ray = kwargs['use_ray']
        self._init_update_steps()

    def _init_update_steps(self):
        if (not self.cfg.on_policy) and self.use_ray:
            self.n_update_steps = float('inf')
        elif self.cfg.on_policy and self.use_ray:
            self.n_update_steps = self.cfg.n_interactors
        else:
            self.n_update_steps = 1

    def run(self):
        run_step = 0
        while True:
            training_data = exec_method(self.collector, 'pub_msg', True, Msg(type = MsgType.COLLECTOR_GET_TRAINING_DATA))
            if training_data is not None:
                self.policy.learn(**training_data)
                global_update_step = exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_GET_UPDATE_STEP))
                exec_method(self.tracker, 'pub_msg', False, Msg(type = MsgType.TRACKER_INCREASE_UPDATE_STEP))
                # put updated model params to policy_mgr
                model_params = self.policy.get_model_params()
                exec_method(self.policy_mgr, 'pub_msg', False, Msg(type = MsgType.MODEL_MGR_PUT_MODEL_PARAMS, data = (global_update_step, model_params)))
                # put policy summary to recorder
                if global_update_step % self.cfg.policy_summary_fre == 0:
                    policy_summary = [(global_update_step,self.policy.get_summary())]
                    exec_method(self.recorder, 'pub_msg', False, Msg(type = MsgType.RECORDER_PUT_SUMMARY, data = policy_summary))
            run_step += 1
            if run_step >= self.n_update_steps:
                return
    
class LearnerMgr(Moduler):
    ''' learner manager
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.policy = kwargs['policy']
        self.recorder = create_module(Recorder, self.use_ray, {'num_cpus':0}, self.cfg, type = 'learner')
        self.learners = [create_module(Learner, self.use_ray, {'num_cpus':1}, 
                                       self.cfg, 
                                       id = i, 
                                       policy = copy.deepcopy(self.policy), 
                                       policy_mgr = kwargs.get('policy_mgr', None), 
                                       collector = kwargs.get('collector', None), 
                                       tracker = kwargs.get('tracker', None), 
                                       recorder = self.recorder,
                                       logger = self.logger,
                                       use_ray = self.use_ray,
                                       )      
                         for i in range(self.cfg.n_learners)]
        exec_method(self.logger, 'info', True, f"[LearnerMgr] Create {self.cfg.n_learners} learners!, use_ray: {self.use_ray}")
        
    def run(self):
        need_get = False
        if self.cfg.on_policy and self.use_ray:
            need_get = True
        for i in range(self.cfg.n_learners):
            exec_method(self.learners[i], 'run', need_get)
