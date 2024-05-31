#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-05-31 11:33:13
Discription: 
'''
import gymnasium as gym
import ray
import time
import copy
from joyrl.algos.base.experience import Exp
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.recorder import Recorder
from joyrl.utils.utils import exec_method, create_module

class Interactor(Moduler):
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.id = kwargs.get('id', 0)
        self.env = kwargs.get('env', None)
        self.policy = copy.deepcopy(kwargs['policy'])
        self._raw_exps_que = kwargs.get('raw_exps_que', None)
        self.data_handler = kwargs['data_handler']
        self.tracker = kwargs['tracker']
        self.collector = kwargs['collector']
        self.recorder = kwargs['recorder']
        self.policy_mgr = kwargs['policy_mgr']
        self._latest_model_params_dict = kwargs.get('latest_model_params_dict', None)
        self.seed = self.cfg.seed + self.id
        self.exps = [] # reset experiences
        self.summary = [] # reset summary
        self.ep_reward, self.ep_step = 0, 0 # reset params per episode
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
        self._update_policy()
        self._init_n_sample_steps()

    def _init_n_sample_steps(self):
        ''' when learner_mode is serial, learner will run after interact finishes n_sample_steps
        '''
        if self.cfg.on_policy:
            # self.n_sample_steps = self.cfg.batch_size // (self.cfg.n_interactors - 1)
            self.n_sample_steps = 200
        else:
            self.n_sample_steps = float('inf')
                
    def _put_exps(self):
        ''' put exps to collector
        '''
        while True:
            try:
                self.exps = exec_method(self.data_handler, 'handle_exps_after_interact', 'get', self.exps)
                self._raw_exps_que.put(self.exps, block=True, timeout=0.1)
                break
            except:
                # exec_method(self.logger, 'warning', 'get', "[Interactor._put_exps] raw_exps_que is full!")
                time.sleep(0.1)
        self.exps = []

    def _update_policy(self):
        ''' update policy
        '''
        model_params_dict = exec_method(self._latest_model_params_dict, 'get_value', 'get') # get model params
        model_params = model_params_dict['model_params']
        model_step = model_params_dict['step']
        # exec_method(self.logger, 'info', 'get', f"Interactor {self.id} update policy with model step {model_step}")
        self.policy.put_model_params(model_params)

    def run(self):
        ''' run in sync mode
        '''
        run_step = 0 # local run step
        while True:
            action = self.policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            self.exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or truncated or self.ep_step >= self.cfg.max_step > 0:
                exec_method(self.tracker, 'pub_msg', 'get', Msg(MsgType.TRACKER_INCREASE_EPISODE))
                global_episode = exec_method(self.tracker, 'pub_msg', 'get', Msg(type = MsgType.TRACKER_GET_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0: 
                    exec_method(self.logger, 'info', 'remote', f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps, truncated: {truncated}, terminated: {terminated}")
                    # put summary to recorder
                    interact_summary = {'reward': self.ep_reward,'step': self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                    exec_method(self.recorder, 'pub_msg', 'remote', Msg(type = MsgType.RECORDER_PUT_SUMMARY, data = self.summary)) # put summary to stats recorder
                    self.summary = [] # reset summary
                self.ep_reward, self.ep_step = 0, 0
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)      
            if len(self.exps) >= self.cfg.exps_trucation_size or terminated or truncated:
                self._update_policy() 
                self._put_exps()    
            run_step += 1
            if run_step >= self.n_sample_steps:
                break