#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2024-01-03 17:30:51
Discription: 
'''
import ray 
from ray.util.queue import Queue as RayQueue
from pathlib import Path
import pickle
import time
import threading

import pandas
from multiprocessing import Queue
from torch.utils.tensorboard import SummaryWriter  
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.utils.utils import exec_method

class Recorder(Moduler):
    ''' Recorder for recording training information
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self._init_writter()
        self._summary_que_dict = {}
        self._summary_que_dict['interact'] = RayQueue(maxsize = 256) if self.use_ray else Queue(maxsize = 256)
        self._summary_que_dict['policy'] = RayQueue(maxsize = 256) if self.use_ray else Queue(maxsize = 256)
        self._t_start()

    def _t_start(self):
        exec_method(self.logger, 'info', False, "Start recorder!")
        self._t_save_interact_summary = threading.Thread(target=self._save_interact_summary)
        self._t_save_interact_summary.setDaemon(True)
        self._t_save_policy_summary = threading.Thread(target=self._save_policy_summary)
        self._t_save_policy_summary.setDaemon(True)
        self._t_save_interact_summary.start()
        self._t_save_policy_summary.start()
        
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.RECORDER_PUT_INTERACT_SUMMARY:
            interact_summary_list = msg_data
            self._add_summary(interact_summary_list, writter_type = 'interact')
        elif msg_type == MsgType.RECORDER_PUT_POLICY_SUMMARY:
            policy_summary_list = msg_data
            self._add_summary(policy_summary_list, writter_type = 'policy')
        else:
            raise NotImplementedError
        
    def _init_writter(self):
        self.writters = {}
        self.writter_types = ['interact','policy']
        for writter_type in self.writter_types:
            self.writters[writter_type] = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/{writter_type}")
    
    def _add_summary(self, summary_data_list, writter_type = None):
        while not self._summary_que_dict[writter_type].full():
            self._summary_que_dict[writter_type].put(summary_data_list)
            time.sleep(0.001)
            break

    def _write_tb_scalar(self, step, summary, writter_type):
        for key, value in summary.items():
            self.writters[writter_type].add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = step)

    def _write_dataframe(self, step, summary, writter_type):
        df_file = f"{self.cfg.res_dir}/{writter_type}.csv"
        if Path(df_file).exists():
            df = pandas.read_csv(df_file)
        else:
            df = pandas.DataFrame()
        saved_dict = {f"{writter_type}_step": step}
        saved_dict.update(summary)
        df = df.append(saved_dict, ignore_index=True)
        df.to_csv(df_file, index = False)

    def _save_interact_summary(self):
        while True:
            try:
                while not self._summary_que_dict['interact'].empty():
                    summary_data_list = self._summary_que_dict['interact'].get()
                    for summary_data in summary_data_list:
                        step, summary = summary_data
                        self._write_tb_scalar(step, summary, writter_type = 'interact')
                        self._write_dataframe(step, summary, writter_type = 'interact')
                    break
            except Exception as e:
                break

    def _save_policy_summary(self):
        while True:
            try:
                while not self._summary_que_dict['policy'].empty():
                    summary_data_list = self._summary_que_dict['policy'].get()
                    for summary_data in summary_data_list:
                        step, summary = summary_data
                        self._write_tb_scalar(step, summary, writter_type = 'policy')
                        self._write_dataframe(step, summary, writter_type = 'policy')
                    break
            except Exception as e:
                break       

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