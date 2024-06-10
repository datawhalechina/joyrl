#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-06-10 21:23:21
Discription: 
'''
import copy
from ray.util.queue import Full
from joyrl.algos.base.experience import Exp
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.recorder import Recorder
from joyrl.framework.utils import exec_method


class Interactor(Moduler):
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.id = kwargs.get('id', 0)
        self.env = kwargs.get('env', None)
        self.policy = copy.deepcopy(kwargs['policy'])
        self.data_handler = kwargs['data_handler']
        self.tracker = kwargs['tracker']
        self.collector = kwargs['collector']
        self.recorder = kwargs['recorder']
        self.policy_mgr = kwargs['policy_mgr']
        self._latest_model_params_dict = kwargs.get('latest_model_params_dict', None)
        self.sample_data_que = kwargs['sample_data_que']
        self.seed = self.cfg.seed + self.id
        self.exps = [] # reset experiences
        self.summary = [] # reset summary
        self.ep_reward, self.ep_step = 0, 0 # reset params per episode
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
        self.truncated, self.terminated = False, False
        self.curr_model_step, self.last_model_step = 0, 0
        self.need_update_policy = False
        self._init_n_sample_steps()
        exec_method(self.logger, 'info', 'remote', f"[Interactor.__init__] Start interactor {self.id}!")

    def _init_n_sample_steps(self):
        ''' when learner_mode is serial, learner will run after interact finishes n_sample_steps
        '''
        if not self.cfg.is_learner_async:
            self.n_sample_steps = self.cfg.exps_trucation_size
        else:
            self.n_sample_steps = float('inf')
                
    def _put_exps(self):
        ''' put exps to collector
        '''
        if len(self.exps) >= self.cfg.exps_trucation_size or self.terminated or self.truncated:
            if self.cfg.is_learner_async:
                exec_method(self.sample_data_que, 'put', 'get', self.exps)
                # if self.sample_data_que.full():
                #     self.sample_data_que.get_nowait()
                # # self.sample_data_que.put(self.exps, block=True, timeout=0.1)
                # try:
                #     self.sample_data_que.put(self.exps)
                # except Full:
                #     # exec_method(self.logger, 'info', 'remote', f"Interactor {self.id} put exps to sample_data_que failed")
                #     pass
            else:
                exec_method(self.collector, 'pub_msg', 'remote', Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps))
            self.exps = []
            self.need_update_policy = True
        else:
            self.need_update_policy = False

    def _update_policy(self):
        ''' update policy
        '''
        if not self.need_update_policy:
            return 

        # while True:
        #     model_params_dict = exec_method(self._latest_model_params_dict, 'get_value', 'get') # get model params
        #     model_params = model_params_dict['model_params']
        #     self.curr_model_step = model_params_dict['step']
        #     if self.curr_model_step > self.last_model_step:
        #         self.last_model_step = self.curr_model_step
        #         # exec_method(self.logger, 'info', 'get', f"Interactor {self.id} update policy with model step {self.curr_model_step}")
        #         self.policy.put_model_params(model_params)
        #         break
        #     time.sleep(0.1)
        
        model_params_dict = exec_method(self._latest_model_params_dict, 'get_value', 'get') # get model params
        model_params = model_params_dict['model_params']
        self.curr_model_step = model_params_dict['step']
        # print(f"[Interactor._update_policy] interactor {self.id} update policy with model step {self.curr_model_step}")
        self.policy.put_model_params(model_params)
        
    def run(self):
        ''' run in sync mode
        '''
        run_step = 0 # local run step
        while True:
            self._update_policy()
            action = self.policy.get_action(self.curr_obs)
            obs, reward, self.terminated, self.truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'model_step': self.curr_model_step, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': self.terminated or self.truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            self.exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if self.terminated or self.truncated or self.ep_step >= self.cfg.max_step > 0:
                exec_method(self.tracker, 'pub_msg', 'remote', Msg(MsgType.TRACKER_INCREASE_EPISODE))
                global_episode = exec_method(self.tracker, 'pub_msg', 'get', Msg(type = MsgType.TRACKER_GET_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0: 
                    exec_method(self.logger, 'info', 'remote', f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps, truncated: {self.truncated}, terminated: {self.terminated}")
                    # put summary to recorder
                    interact_summary = {'reward': self.ep_reward,'step': self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                    exec_method(self.recorder, 'pub_msg', 'remote', Msg(type = MsgType.RECORDER_PUT_SUMMARY, data = self.summary)) # put summary to stats recorder
                    self.summary = [] # reset summary
                self.ep_reward, self.ep_step = 0, 0
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)      
            self._put_exps()
            run_step += 1
            if run_step >= self.n_sample_steps:
                break