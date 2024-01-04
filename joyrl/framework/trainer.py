#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-04 13:25:24
Discription: 
'''
import time
import ray
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.utils.utils import print_logs, exec_method

class Trainer(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig,**kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.policy_mgr = kwargs['policy_mgr']
        self.interactor_mgr = kwargs['interactor_mgr']
        self.learner_mgr = kwargs['learner_mgr']
        self.collector = kwargs['collector']
        self.tracker = kwargs['tracker']
        self.recorder = kwargs['recorder']
        self._print_cfgs() # print parameters

    def _print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            print_logs(self.logger, f"{name}:", is_ray = self.use_ray)
            print_logs(self.logger, ''.join(['='] * 80), is_ray = self.use_ray)
            tplt = "{:^20}\t{:^20}\t{:^20}"
            print_logs(self.logger, tplt.format("Name", "Value", "Type"), is_ray = self.use_ray)
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                print_logs(self.logger, tplt.format(k, v, str(type(v))), is_ray = self.use_ray)
            print_logs(self.logger, ''.join(['='] * 80), is_ray = self.use_ray)
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')

    def run(self):
        ''' run the trainer
        '''
        self.logger.info(f"[Trainer.run] Start {self.cfg.mode}ing!") # print info
        s_t = time.time()
        while True:
            self.interactor_mgr.run()
            if self.cfg.mode.lower() == 'train':
                self.learner_mgr.run()
            if exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                e_t = time.time()
                self.logger.info(f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                time.sleep(3)
                break

    def ray_run(self):
        if self.cfg.online_eval:
            self.online_tester.init.remote()
        self.policy_mgr.init.remote()
        self.recorder.init.remote()
        self.collector.init.remote()
        self.interactor_mgr.init.remote(
            policy_mgr = self.policy_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.learner_mgr.init.remote(
            policy_mgr = self.policy_mgr,
            tracker = self.tracker,
            collector = self.collector,
            recorder = self.recorder,
            logger = self.logger
        )
        self.logger.info.remote(f"[Trainer.ray_run] Start {self.cfg.mode}ing!") # print info
        self.interactor_mgr.ray_run.remote()
        self.learner_mgr.ray_run.remote()
        s_t = time.time()
        while True:
            if ray.get(self.tracker.pub_msg.remote(Msg(type = MsgType.TRACKER_CHECK_TASK_END))):
                e_t = time.time()
                self.logger.info.remote(f"[Trainer.ray_run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                time.sleep(10)
                ray.shutdown()
                break 
