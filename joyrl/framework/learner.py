#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2023-12-25 00:48:52
Discription: 
'''
import ray
import copy
import time
from typing import Tuple
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler

class Learner:
    ''' learner
    '''
    def __init__(self, cfg : MergedConfig, id = 0, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
        self.model_mgr = kwargs['model_mgr']
        self.collector = kwargs['collector']
        self.tracker = kwargs['tracker']
        self.logger = kwargs['logger']
        self.recorder = kwargs['recorder']
    
    def run(self):
        training_data = self.collector.pub_msg(Msg(type = MsgType.COLLECTOR_GET_TRAINING_DATA))
        if training_data is None: return
        self.policy.learn(**training_data)
        global_update_step = self.tracker.pub_msg(Msg(type = MsgType.TRACKER_GET_UPDATE_STEP))
        self.tracker.pub_msg(Msg(type = MsgType.TRACKER_INCREASE_UPDATE_STEP))
        # put updated model params to model_mgr
        model_params = self.policy.get_model_params()
        self.model_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_PUT_MODEL_PARAMS, data = (global_update_step, model_params)))
        # put policy summary to recorder
        if global_update_step % self.cfg.policy_summary_fre == 0:
            policy_summary = [(global_update_step,self.policy.get_summary())]
            self.recorder.pub_msg(Msg(type = MsgType.RECORDER_PUT_POLICY_SUMMARY, data = policy_summary))

    def ray_run(self):
        while True:
            training_data = ray.get(self.collector.pub_msg.remote(Msg(type = MsgType.COLLECTOR_GET_TRAINING_DATA)))
            if training_data is None: continue
            self.policy.learn(**training_data)
            global_update_step = ray.get(self.tracker.pub_msg.remote(Msg(type = MsgType.TRACKER_GET_UPDATE_STEP)))
            self.tracker.pub_msg.remote(Msg(type = MsgType.TRACKER_INCREASE_UPDATE_STEP))
            # put updated model params to model_mgr
            model_params = self.policy.get_model_params()
            self.model_mgr.pub_msg.remote(Msg(type = MsgType.MODEL_MGR_PUT_MODEL_PARAMS, data = (global_update_step, model_params)))

            # put policy summary to recorder
            if global_update_step % self.cfg.policy_summary_fre == 0:
                s_t = time.time()
                policy_summary = [(global_update_step,self.policy.get_summary())]
                self.recorder.pub_msg.remote(Msg(type = MsgType.RECORDER_PUT_POLICY_SUMMARY, data = policy_summary))
                e_t = time.time()
                # self.logger.info.remote(f"Put policy summary finished in {e_t - s_t:.3f} s")

    def _get_id(self):
        return self.id
    
class LearnerMgr(Moduler):
    ''' learner manager
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.policy = kwargs['policy']

    def init(self, *args, **kwargs):
        if self.use_ray:
            self.learners = [ray.remote(Learner).options(num_cpus=3).remote(cfg = self.cfg, id = i, policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.cfg.n_learners)]
        else:
            self.learners = [Learner(cfg = self.cfg, id = i, policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.cfg.n_learners)]

    def run(self):
        for i in range(self.cfg.n_learners):
            self.learners[i].run()

    def ray_run(self):
        for i in range(self.cfg.n_learners):
            self.learners[i].ray_run.remote()
