#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-02 15:02:30
LastEditor: JiangJi
LastEditTime: 2024-06-14 17:50:26
Discription: 
'''
import time
import copy
import os
import threading
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.utils import exec_method, load_model_meta, save_model_meta
    
class OnlineTester(Moduler):
    ''' Online tester
    '''
    def __init__(self, cfg : MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.env = copy.deepcopy(kwargs['env'])
        self.policy = copy.deepcopy(kwargs['policy'])
        self.recorder = kwargs['recorder']
        self.policy_mgr = kwargs['policy_mgr']
        self.tracker = kwargs['tracker']
        self.seed = self.cfg.seed
        self.best_eval_reward = -float('inf')
        self.curr_test_step = -1
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
        self.reward_threshold_cnt = 0
        self._t_start()

    def _t_start(self):
        exec_method(self.logger, 'info', 'remote', "[OnlineTester._t_start] Start online tester!")
        self._t_eval_policy = threading.Thread(target=self._eval_policy)
        self._t_eval_policy.setDaemon(True)
        self._t_eval_policy.start()
    
    def _check_updated_model(self):
        model_step_list = os.listdir(self.cfg.model_dir)
        model_step_list = [int(model_step) for model_step in model_step_list if model_step.isdigit()]
        model_step_list.sort()
        if len(model_step_list) == 0:
            return False, -1
        elif model_step_list[-1] == self.curr_test_step:
            return False, -1
        elif model_step_list[-1] > self.curr_test_step:
            return True, model_step_list[-1]
        
    def _eval_policy(self):
        ''' Evaluate policy
        '''
        while True:
            updated, model_step = self._check_updated_model()
            if updated:
                self.curr_test_step = model_step
                self.policy.load_model(f"{self.cfg.model_dir}/{self.curr_test_step}")
                sum_eval_reward = 0
                for _ in range(self.cfg.online_eval_episode):
                    ep_reward, ep_step = 0, 0
                    while True:
                        action = self.policy.get_action(self.curr_obs, mode = 'predict')
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        self.curr_obs, self.curr_info = obs, info
                        ep_reward += reward
                        ep_step += 1
                        if terminated or truncated or (0 <= self.cfg.max_step <= ep_step):
                            sum_eval_reward += ep_reward
                            self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)
                            break
                try:
                    self.env.close()
                except:
                    pass
                mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
                exec_method(self.logger, 'info', 'get', f"online_eval step: {self.curr_test_step}, online_eval_reward: {mean_eval_reward:.3f}")
                exec_method(self.recorder, 'pub_msg', 'remote', Msg(type = MsgType.RECORDER_PUT_SUMMARY, data = [(model_step, {'online_eval_reward': mean_eval_reward})])) # put summary to stats recorder
                # logger_info = f"test_step: {self.curr_test_step}, online_eval_reward: {mean_eval_reward:.3f}"
                # self.logger.info.remote(logger_info) if self.use_ray else self.logger.info(logger_info)
                if mean_eval_reward >= self.best_eval_reward:
                    exec_method(self.logger, 'info', 'get', f"current online_eval step obtain a better reward: {mean_eval_reward:.3f}, save the best model!")
                    self.policy.save_model(f"{self.cfg.model_dir}/best")
                    self.best_eval_reward = mean_eval_reward
                    model_meta = {'best_eval_reward': self.best_eval_reward, 'best_model_step': model_step}
                    exec_method(self.policy_mgr, 'pub_msg', 'remote', Msg(type = MsgType.POLICY_MGR_PUT_MODEL_META, data = (self.name, model_meta)))
                if mean_eval_reward >= self.cfg.reward_threshold:
                    self.reward_threshold_cnt += 1
                    if self.reward_threshold_cnt >= self.cfg.reward_threshold_limit:
                        exec_method(self.logger, 'info', 'remote', f"[OnlineTester._eval_policy] policy has reached the reward threshold: {self.cfg.reward_threshold}, over {self.cfg.reward_threshold_limit} episodes, will stop training!")
                        exec_method(self.tracker, 'pub_msg', 'remote', Msg(type = MsgType.TRACKER_FORCE_TASK_END))
            time.sleep(1)
        
