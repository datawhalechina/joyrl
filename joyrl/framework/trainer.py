#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-06-01 16:53:24
Discription: 
'''
import copy
import time
import ray
from ray.util.queue import Queue as RayQueue
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.collector import Collector
from joyrl.framework.tracker import Tracker
from joyrl.framework.interactor import Interactor
from joyrl.framework.learner import Learner
from joyrl.framework.tester import OnlineTester
from joyrl.framework.policy_mgr import PolicyMgr
from joyrl.framework.recorder import Recorder
from joyrl.utils.utils import exec_method, create_module
from joyrl.utils.utils import Logger
from joyrl.framework.utils import SharedData

class Trainer(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig,**kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.env = kwargs['env']
        self.policy = kwargs['policy']
        self.data_handler = kwargs['data_handler']
        self._print_cfgs() # print parameters
        self._create_shared_data() # create data queues
        self._create_modules() # create modules
        
    def _create_shared_data(self):
        self.latest_model_params_dict = SharedData.options(**{'num_cpus': 0}).remote({'step': 0, 'model_params': self.policy.get_model_params()})
        
    def _create_modules(self):
        ''' create modules
        '''
        if self.cfg.online_eval:
            recorder = ray.remote(Recorder).options(**{'num_cpus': 0}).remote(self.cfg,
                                                                            name = 'RecorderOnlineTester',
                                                                            type = 'online_tester')
            self.online_tester = OnlineTester(
                self.cfg, 
                name = 'OnlineTester',
                env = copy.deepcopy(self.env), 
                policy = copy.deepcopy(self.policy),
                recorder = recorder,
            )
        self.tracker = ray.remote(Tracker).remote(self.cfg)
        self.collector = ray.remote(Collector).options(**{'num_cpus': 1}).remote(
            self.cfg,
            name = 'Collector',
            data_handler = self.data_handler,
        )
        self.policy_mgr = ray.remote(PolicyMgr).options(**{'num_cpus': 0}).remote(
            self.cfg,
            name = 'PolicyMgr',
            policy = copy.deepcopy(self.policy),
            latest_model_params_dict = self.latest_model_params_dict,
        )
        self.interactors = []
        recorder = ray.remote(Recorder).options(**{'num_cpus': 0}).remote(self.cfg,
                                                                            name = 'RecorderInteractor',
                                                                            type = 'interactor')
        for i in range(self.cfg.n_interactors):
            interactor = ray.remote(Interactor).options(**{'num_cpus': 1}).remote(
                self.cfg,
                id = i,
                name = f"Interactor_{i}",
                env = copy.deepcopy(self.env),
                policy = copy.deepcopy(self.policy),
                data_handler = copy.deepcopy(self.data_handler),  # only use the static method handle_exps_after_interact
                tracker = self.tracker,
                collector = self.collector,
                recorder = recorder,
                policy_mgr = self.policy_mgr,
                latest_model_params_dict = self.latest_model_params_dict,
                )
            self.interactors.append(interactor)
        self.learners = []
        recorder = ray.remote(Recorder).options(**{'num_cpus': 0}).remote(self.cfg,
                                                                            name = 'RecorderLearner',
                                                                            type = 'learner')
        for i in range(self.cfg.n_learners):
            learner = ray.remote(Learner).remote(
                self.cfg,
                id = i,
                name = f"Learner_{i}",
                policy = copy.deepcopy(self.policy),
                policy_mgr = self.policy_mgr,
                collector = self.collector,
                data_handler = self.data_handler,
                tracker = self.tracker,
                recorder = recorder,
                )
            self.learners.append(learner)

    def _print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            exec_method(self.logger, 'info', 'get', f"{name}:")
            exec_method(self.logger, 'info', 'get', ''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            exec_method(self.logger, 'info', 'get', tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                exec_method(self.logger, 'info', 'get', tplt.format(k, v, str(type(v))))
            exec_method(self.logger, 'info', 'get', ''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')

    def run(self):
        ''' run the trainer
        '''
        exec_method(self.logger, 'info', 'get', f"[Trainer.run] Start {self.cfg.mode}ing!")
        s_t = time.time()
        if not self.cfg.is_learner_async:
            while True:
                ray.get([interactor.run.remote() for interactor in self.interactors])
                ray.get([learner.run.remote() for learner in self.learners])
                if exec_method(self.tracker, 'pub_msg', 'get', Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                    e_t = time.time()
                    exec_method(self.logger, 'info', 'get', f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                    time.sleep(5)
                    ray.shutdown()
                    break
        else:
            [interactor.run.remote() for interactor in self.interactors]
            [learner.run.remote() for learner in self.learners]
            while True:
                if exec_method(self.tracker, 'pub_msg', 'get', Msg(type = MsgType.TRACKER_CHECK_TASK_END)):
                    e_t = time.time()
                    exec_method(self.logger, 'info', 'get', f"[Trainer.run] Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s")
                    time.sleep(5)
                    ray.shutdown()
                    break
                time.sleep(1)