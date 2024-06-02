#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2024-06-02 10:50:37
Discription: 
'''
import ray 
from ray.util.queue import Queue as RayQueue
from pathlib import Path
import pickle
import time
import threading

import pandas as pd
from queue import Queue
from torch.utils.tensorboard import SummaryWriter  
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.utils import exec_method

class Recorder(Moduler):
    ''' Recorder for recording training information
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.type = kwargs.get('type', 'recorder')
        self.writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/{self.type}")
        self._summary_que = RayQueue(maxsize = 256)
        self._t_start() # TODO, slow down the training speed when using threading 

    def _t_start(self):
        exec_method(self.logger, 'info', 'remote', f"[Recorder._t_start] Start {self.type} recorder!")
        self._t_save_summary = threading.Thread(target=self._save_summary)
        self._t_save_summary.setDaemon(True)
        self._t_save_summary.start()

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.RECORDER_PUT_SUMMARY:
            summary_data_list = msg_data
            self._add_summary(summary_data_list)
        else:
            raise NotImplementedError
        
    def _init_writter(self):
        self.writters = {}
        self.writter_types = ['interact','policy']
        for writter_type in self.writter_types:
            self.writters[writter_type] = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/{writter_type}")
    
    def _add_summary(self, summary_data_list):
        while True:
            try:
                self._summary_que.put(summary_data_list, block = False)
                break
            except:
                self.logger.warning(f"[Recorder._add_summary] {self.type} summary_que is full!")
                # time.sleep(0.001)
                pass

    def _write_tb_scalar(self, step: int, summary: dict):
        for key, value in summary.items():
            self.writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value = value, global_step = step)

    def _write_dataframe(self, step: int, summary: dict):
        df_file = f"{self.cfg.res_dir}/{self.type}.csv"
        if Path(df_file).exists():
            df = pd.read_csv(df_file)
        else:
            df = pd.DataFrame()
        saved_dict = {f"{self.type}_step": step}
        saved_dict.update(summary)
        df = pd.concat([df, pd.DataFrame(saved_dict, index=[0])], ignore_index = True)
        df.to_csv(df_file, index = False)

    def _save_summary(self):
        while True:
            while not self._summary_que.empty():
                summary_data_list = self._summary_que.get()
                for summary_data in summary_data_list:
                    step, summary = summary_data
                    self._write_tb_scalar(step, summary)
                    self._write_dataframe(step, summary)
            time.sleep(0.001)

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