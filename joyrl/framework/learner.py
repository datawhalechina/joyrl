#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-03 13:45:43
Discription: 
'''
import ray
import copy
import time
from typing import Tuple
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler

class Learner(Moduler):
    ''' learner
    '''
    def __init__(self, cfg : MergedConfig, **kwargs) -> None:
        super().__init__(cfg,**kwargs)
        self.id = kwargs.get('id', 0)
        self.policy = kwargs.get('policy', None)
        self.policy_mgr = kwargs.get('policy_mgr', None)
        self.collector = kwargs.get('collector', None)
        self.tracker = kwargs.get('tracker', None)
        self.recorder = kwargs.get('recorder', None)
        self._init_update_steps()

    def _init_update_steps(self):
        if not self.cfg.on_policy and self.use_ray:
            self.n_update_steps = float('inf')
        else:
            self.n_update_steps = 1

    def run(self):
        run_step = 0
        while True:
            training_data = self.collector.pub_msg(Msg(type = MsgType.COLLECTOR_GET_TRAINING_DATA))
            if training_data is None: return
            self.policy.learn(**training_data)
            global_update_step = self.tracker.pub_msg(Msg(type = MsgType.TRACKER_GET_UPDATE_STEP))
            self.tracker.pub_msg(Msg(type = MsgType.TRACKER_INCREASE_UPDATE_STEP))
            # put updated model params to policy_mgr
            model_params = self.policy.get_model_params()
            self.policy_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_PUT_MODEL_PARAMS, data = (global_update_step, model_params)))
            # put policy summary to recorder
            if global_update_step % self.cfg.policy_summary_fre == 0:
                policy_summary = [(global_update_step,self.policy.get_summary())]
                self.recorder.pub_msg(Msg(type = MsgType.RECORDER_PUT_POLICY_SUMMARY, data = policy_summary))
            run_step += 1
            if run_step >= self.n_update_steps:
                return
    
class LearnerMgr(Moduler):
    ''' learner manager
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.policy = kwargs['policy']
        self.learners = [Learner(cfg = self.cfg, 
                                 id = i, 
                                 policy = copy.deepcopy(self.policy),
                                 policy_mgr = kwargs.get('policy_mgr', None),
                                 collector = kwargs.get('collector', None),
                                 tracker = kwargs.get('tracker', None),
                                 recorder = kwargs.get('recorder', None),
                                 ) for i in range(self.cfg.n_learners)]

    # def init(self, *args, **kwargs):
    #     if self.use_ray:
    #         self.learners = [ray.remote(Learner).options(num_cpus=3).remote(cfg = self.cfg, id = i, policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.cfg.n_learners)]
    #     else:
    #         self.learners = [Learner(cfg = self.cfg, id = i, policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.cfg.n_learners)]

    def run(self):
        for i in range(self.cfg.n_learners):
            self.learners[i].run()

    def ray_run(self):
        for i in range(self.cfg.n_learners):
            self.learners[i].ray_run.remote()
