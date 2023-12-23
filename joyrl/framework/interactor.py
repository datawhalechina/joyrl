import gymnasium as gym
import ray
import copy
import time
import multiprocessing as mp
from typing import Tuple
from joyrl.algos.base.exps import Exp
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.framework.base import Moduler

class Interactor:
    def __init__(self, cfg: MergedConfig, id = 0, env = None, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg 
        self.tracker = kwargs['tracker']
        self.collector = kwargs['collector']
        self.recorder = kwargs['recorder']
        self.model_mgr = kwargs['model_mgr']
        self.logger = kwargs['logger']
        self.id = id
        self.env = env
        self.policy = policy
        self.seed = self.cfg.seed + self.id
        self.exps = []
        self.seed = self.cfg.seed + self.id
        self.exps = [] # reset experiences
        self.summary = [] # reset summary
        self.ep_reward, self.ep_step = 0, 0 # reset params per episode
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
        self._init_n_sample_steps()

    def _init_n_sample_steps(self):
        ''' when learner_mode is serial, learner will run after interact finishes n_sample_steps
        '''
        if self.cfg.buffer_type.lower().startswith('onpolicy'):
            self.n_sample_steps = self.cfg.batch_size 
        else:
            self.n_sample_steps = 1

    def run(self):
        ''' run in sync mode
        '''
        run_step = 0 # local run step
        model_params = self.model_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_GET_MODEL_PARAMS)) # get model params
        self.policy.put_model_params(model_params)
        while True:
            action = self.policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            self.exps.append(Exp(**interact_transition, **policy_transition))
            run_step += 1
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or truncated or self.ep_step >= self.cfg.max_step:
                global_episode = self.tracker.pub_msg(Msg(MsgType.TRACKER_GET_EPISODE))
                self.tracker.pub_msg(Msg(MsgType.TRACKER_INCREASE_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0: 
                    self.logger.info(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    # put summary to recorder
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                    self.recorder.pub_msg(Msg(type = MsgType.RECORDER_PUT_INTERACT_SUMMARY, data = self.summary))
                    self.summary = [] # reset summary
                self.ep_reward, self.ep_step = 0, 0
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)
            if run_step >= self.n_sample_steps:
                # put exps to collector
                self.collector.pub_msg(Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps))
                self.exps = []
                break

    def ray_run(self):
        ''' start in async mode
        '''
        while True:
            model_params = ray.get(self.model_mgr.pub_msg.remote(Msg(type = MsgType.MODEL_MGR_GET_MODEL_PARAMS))) # get model params
            self.policy.put_model_params(model_params)
            action = self.policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            # create exp
            self.exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if len(self.exps) >= 1 or terminated or truncated or self.ep_step >= self.cfg.max_step:
                self.collector.pub_msg.remote(Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps))
                self.exps = []
            if terminated or truncated or self.ep_step >= self.cfg.max_step:
                global_episode = ray.get(self.tracker.pub_msg.remote(Msg(type = MsgType.TRACKER_GET_EPISODE)))
                self.tracker.pub_msg.remote(Msg(MsgType.TRACKER_INCREASE_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0: 
                    self.logger.info.remote(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                    self.recorder.pub_msg.remote(Msg(type = MsgType.RECORDER_PUT_INTERACT_SUMMARY, data = self.summary)) # put summary to stats recorder
                    self.summary = [] # reset summary
                self.ep_reward, self.ep_step = 0, 0
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)

class InteractorMgr(Moduler):
    ''' Interactor manager for managing interactors
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.env = kwargs['env']
        self.policy = kwargs['policy']
        self.logger = kwargs['logger']
        self.n_interactors = self.cfg.n_interactors
    
    def init(self, *args, **kwargs):
        if self.use_ray:
            self.interactors = [ray.remote(Interactor).options(num_cpus=1).remote(self.cfg, id = i, env = copy.deepcopy(self.env), policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.n_interactors)]
        else:
            self.interactors = [Interactor(self.cfg, id = i, env = copy.deepcopy(self.env), policy = copy.deepcopy(self.policy), *args, **kwargs) for i in range(self.n_interactors)]
    
    def run(self):
        if self.cfg.interactor_mode == 'dummy':
            for i in range(self.n_interactors):
                self.interactors[i].run()
        else:
            raise NotImplementedError(f"[InteractorMgr.run] interactor_mode {self.cfg.interactor_mode} is not implemented!")
            
    def ray_run(self): 
        self.logger.info.remote(f"[InteractorMgr.run] Start interactors!")
        for i in range(self.n_interactors):
            self.interactors[i].ray_run.remote()