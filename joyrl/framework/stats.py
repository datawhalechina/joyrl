#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-27 20:32:07
LastEditor: JiangJi
LastEditTime: 2023-05-27 20:48:22
Discription: 
'''
import ray 
from ray.util.queue import Queue, Empty, Full
from pathlib import Path
import pickle
import logging
from torch.utils.tensorboard import SummaryWriter  
@ray.remote
class StatsRecorder:
    ''' statistics recorder
    '''
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.interact_summary_que = Queue(maxsize=128)
        self.model_summary_que = Queue(maxsize=128)
        self.interact_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/interact")
        self.policy_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/model")
    def add_interact_summary(self,summary):
        ''' add interact summary
        '''
        self.interact_summary_que.put(summary, block=False)
        self.write_interact_summary()
    def add_model_summary(self,summary):
        ''' add model summary
        '''
        self.model_summary_que.put(summary, block=False) 
        self.write_model_summary()
    def write_interact_summary(self):
        while self.interact_summary_que.qsize() > 0:
            episode,interact_summary = self.interact_summary_que.get()
            for key, value in interact_summary.items():
                self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = episode)
    def write_model_summary(self):
        while self.model_summary_que.qsize() > 0:
            update_step, model_summary = self.model_summary_que.get()
            for key, value in model_summary['scalar'].items():
                self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
class BaseLogger(object):
    def __init__(self, fpath = None) -> None:
        Path(fpath).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name="BaseLog")  
        self.logger.setLevel(logging.INFO) # default level is INFO
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        # output to file by using FileHandler
        fh = logging.FileHandler(f"{fpath}/log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
    def info(self, msg):
        self.logger.info(msg)

class SimpleLogger(BaseLogger):
    ''' Simple logger for print log to console
    '''
    def __init__(self, fpath = None) -> None:
        super().__init__(fpath)
        self.logger.name = "SimpleLog"
        # output to console by using StreamHandler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

@ray.remote(num_cpus=0)
class RayLogger(BaseLogger):
    ''' Ray logger for print log to console
    '''
    def __init__(self, fpath=None) -> None:
        super().__init__(fpath)
        self.logger.name = "RayLog"
    def info(self, msg):
        super().info(msg)
        print(msg) # print log to console

class BaseTrajCollector:
    ''' Base class for trajectory collector
    '''
    def __init__(self, fpath) -> None:
        pass
class SimpleTrajCollector(BaseTrajCollector):
    ''' Simple trajectory collector for store trajectories
    '''
    def __init__(self, fpath) -> None:
        super().__init__(fpath)
        self.fpath = fpath
        self.traj_num = 0
        self.init_traj()
        self.init_traj_cache()
    def init_traj(self):
        ''' init trajectories
        '''
        self.trajs = {'state':[],'action':[],'reward':[],'next_state':[],'terminated':[],'info':[]}
    def init_traj_cache(self):
        ''' init trajectory cache for one episode
        '''
        self.ep_states, self.ep_actions, self.ep_rewards, self.ep_next_states, self.ep_terminated, self.ep_infos = [], [], [], [], [], []
    def add_traj_cache(self,state,action,reward,next_state,terminated,info):
        ''' store one episode trajectory
        '''
        self.ep_states.append(state)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_next_states.append(next_state)
        self.ep_terminated.append(terminated)
        self.ep_infos.append(info)
    def store_traj(self, task_end_flag = False):
        ''' store trajectory cache into trajectories
        '''
        self.trajs['state'].append(self.ep_states)
        self.trajs['action'].append(self.ep_actions)
        self.trajs['reward'].append(self.ep_rewards)
        self.trajs['next_state'].append(self.ep_next_states)
        self.trajs['terminated'].append(self.ep_terminated)
        self.trajs['info'].append(self.ep_infos)
        if len(self.trajs['state']) >= 1000 or task_end_flag: # save traj when traj number is greater than 1000
            with open(f"{self.fpath}/trajs_{self.traj_num}.pkl", 'wb') as f:
                pickle.dump(self.trajs, f)
                self.traj_num += 1
            self.init_traj_cache()