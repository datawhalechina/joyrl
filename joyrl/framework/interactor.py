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

class BaseInteractor(object):
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        self.cfg = cfg 

class Interactor:
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        self.cfg = cfg
        self.id = kwargs.get('id', 0)
        self.env = kwargs.get('env', None)
        self.policy = kwargs.get('policy', None)
        self._raw_exps_que = kwargs.get('raw_exps_que', None)
        self.tracker = kwargs['tracker']
        self.collector = kwargs['collector']
        self.recorder = kwargs['recorder']
        self.policy_mgr = kwargs['policy_mgr']
        self.logger = kwargs['logger']
        self.use_ray = kwargs['use_ray']
        self.seed = self.cfg.seed + self.id
        self.exps = [] # reset experiences
        self.summary = [] # reset summary
        self.ep_reward, self.ep_step = 0, 0 # reset params per episode
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
        self._init_n_sample_steps()

    def _init_n_sample_steps(self):
        ''' when learner_mode is serial, learner will run after interact finishes n_sample_steps
        '''
        if self.cfg.on_policy:
            self.n_sample_steps = self.cfg.batch_size 
        else:
            self.n_sample_steps = 1
            if self.use_ray: # async mode, set n_sample_steps to inf
                self.n_sample_steps = float('inf')
                
    def _put_exps(self):
        ''' put exps to collector
        '''
        while True:
            try:
                self._raw_exps_que.put(self.exps, block=True, timeout=0.01)
                break
            except:
                exec_method(self.logger, 'warning', True, "[Collector.pub_msg] raw_exps_que is full!")
                time.sleep(0.01)
        self.exps = []

    def run(self):
        ''' run in sync mode
        '''
        run_step = 0 # local run step
        while True:
            model_params = exec_method(self.policy_mgr, 'pub_msg', True,  Msg(type = MsgType.MODEL_MGR_GET_MODEL_PARAMS)) # get model params
            self.policy.put_model_params(model_params)
            action = self.policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            self.exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or truncated or self.ep_step >= self.cfg.max_step > 0:
                global_episode = exec_method(self.tracker, 'pub_msg', True, Msg(type = MsgType.TRACKER_GET_EPISODE))
                exec_method(self.tracker, 'pub_msg', False, Msg(MsgType.TRACKER_INCREASE_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0: 
                    exec_method(self.logger, 'info', True, f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps, truncated: {truncated}, terminated: {terminated}")
                    # put summary to recorder
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                    exec_method(self.recorder, 'pub_msg', True, Msg(type = MsgType.RECORDER_PUT_SUMMARY, data = self.summary)) # put summary to stats recorder
                    self.summary = [] # reset summary
                self.ep_reward, self.ep_step = 0, 0
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)
            if self.cfg.on_policy:
                if len(self.exps) >= self.cfg.batch_size:
                    self._put_exps()
                    # exec_method(self.collector, 'pub_msg', True, Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps))
                    # self.exps = []
            else:
                if len(self.exps) >= self.cfg.exps_trucation_size or terminated or truncated or self.ep_step >= self.cfg.max_step:
                    self._put_exps()
                    # exec_method(self.collector, 'pub_msg', True, Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps))
                    
            run_step += 1
            if run_step >= self.n_sample_steps:
                break

class InteractorMgr(Moduler):
    ''' Interactor manager for managing interactors
    '''
    def __init__(self, cfg: MergedConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.env = kwargs['env']
        self.policy = kwargs['policy']
        self.raw_exps_que = kwargs['raw_exps_que']
        self.recorder = create_module(Recorder, self.use_ray, {'num_cpus':0}, self.cfg, type = 'interactor')
        self.n_interactors = self.cfg.n_interactors
        self.interactors = [create_module(Interactor, self.use_ray, {'num_cpus':1 }, self.cfg,
            id = i,
            env = copy.deepcopy(self.env),
            policy = copy.deepcopy(self.policy),
            tracker = kwargs.get('tracker', None),
            collector = kwargs.get('collector', None),
            recorder = self.recorder,
            policy_mgr = kwargs.get('policy_mgr', None),
            raw_exps_que = self.raw_exps_que,
            logger = self.logger,
            use_ray = self.use_ray,
            ) for i in range(self.n_interactors)
            ]
        exec_method(self.logger, 'info', True, f"[InteractorMgr] Create {self.n_interactors} interactors!, use_ray: {self.use_ray}")

    def run(self):
        ''' run interactors
        '''
        need_get = False
        if self.cfg.on_policy and self.use_ray:
            need_get = True
        for i in range(self.n_interactors):
            exec_method(self.interactors[i], 'run', need_get)
            