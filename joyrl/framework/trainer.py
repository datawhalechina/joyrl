#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-01-07 00:36:09
Discription: 
'''
import time
import ray
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.utils.utils import exec_method

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
        self._print_cfgs() # print parameters

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
                time.sleep(0.1)
                if exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                    e_t = time.time()
                    exec_method(self.logger, 'info', True, f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                    time.sleep(5)
                    ray.shutdown()
                    break
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

    