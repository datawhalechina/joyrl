#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-21 18:01:28
Discription: 
'''
import time
import ray
from queue import Queue
from ray.util.queue import Queue as RayQueue
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.collector import Collector
from joyrl.framework.tracker import Tracker
from joyrl.framework.interactor import InteractorMgr
from joyrl.framework.learner import LearnerMgr
from joyrl.framework.tester import OnlineTester
from joyrl.framework.policy_mgr import PolicyMgr
from joyrl.utils.utils import exec_method, create_module

class Trainer(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig,**kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.env = kwargs['env']
        self.policy = kwargs['policy']
        self.data_handler = kwargs['data_handler']
        self._print_cfgs() # print parameters
        self._create_data_ques() # create data queues
        self._create_modules() # create modules
    def _create_data_ques(self):
        self.raw_exps_que = RayQueue(maxsize = 256) if self.use_ray else Queue(maxsize = 256)
    def _create_modules(self):
        ''' create modules
        '''
        if self.cfg.online_eval:
            self.online_tester = create_module(OnlineTester, False, {'num_cpus':0}, self.cfg, env = self.env, policy = self.policy)
        self.tracker = create_module(Tracker, self.use_ray, {'num_cpus':0}, self.cfg)
        self.collector = create_module(Collector, self.use_ray, {'num_cpus':1},
                                        self.cfg, 
                                        data_handler = self.data_handler,
                                        raw_exps_que = self.raw_exps_que
                                        )
        self.policy_mgr = create_module(PolicyMgr, self.use_ray, {'num_cpus':0}, self.cfg, policy = self.policy)
        self.interactor_mgr = create_module(InteractorMgr, self.use_ray, {'num_cpus':0},
                                            self.cfg, 
                                            env = self.env, 
                                            policy = self.policy, 
                                            collector = self.collector, 
                                            tracker = self.tracker, 
                                            policy_mgr = self.policy_mgr,
                                            raw_exps_que = self.raw_exps_que
                                            )
        self.learner_mgr = create_module(LearnerMgr, self.use_ray, {'num_cpus':0},
                                            self.cfg, 
                                            policy = self.policy,
                                            collector = self.collector,
                                            tracker = self.tracker,
                                            policy_mgr = self.policy_mgr
                                            )

    def _print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            exec_method(self.logger, 'info', True, f"{name}:")
            exec_method(self.logger, 'info', True, ''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            exec_method(self.logger, 'info', True, tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                exec_method(self.logger, 'info', True, tplt.format(k, v, str(type(v))))
            exec_method(self.logger, 'info', True, ''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')

    def run(self):
        ''' run the trainer
        '''
        exec_method(self.logger, 'info', True, f"[Trainer.run] Start {self.cfg.mode}ing!")
        s_t = time.time()
        if (not self.cfg.on_policy) and self.use_ray: # if off-policy, async training
            exec_method(self.interactor_mgr, 'run', False)
            exec_method(self.learner_mgr, 'run', False)
            while True:
                if exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                    e_t = time.time()
                    exec_method(self.logger, 'info', True, f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                    time.sleep(5)
                    ray.shutdown()
                    break
                time.sleep(0.1)
        else: # if on-policy or simple training,  sync training
            while True:
                exec_method(self.interactor_mgr, 'run', True)
                if self.cfg.mode.lower() == 'train':
                    exec_method(self.learner_mgr, 'run', True)
                if exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                    e_t = time.time()
                    exec_method(self.logger, 'info', True, f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                    time.sleep(5)
                    break

    