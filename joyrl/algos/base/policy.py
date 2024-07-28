import copy, dill
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from gymnasium.spaces import Box, Discrete
from joyrl.algos.base.action_layer import ActionLayerType
from joyrl.framework.config import MergedConfig
from joyrl.algos.base.network import BaseNework
class BasePolicy(object):
    ''' base policy for DRL
    '''
    def __init__(self, cfg : MergedConfig, **kwargs) -> None:
        self.cfg = cfg
        self.policy_transition = {}
        self.data_after_train = {}
        self.model_meta = {}
        self.state_size_list = self.cfg.obs_space_info.size
        self.action_size_list = self.cfg.action_space_info.size
        self.to("cpu")
        self.create_model()
        self.create_optimizer()
        self.create_summary()
        # default device is cpu
    
    def to(self, device):
        ''' set device
        '''
        for obj in self.__dict__.values():
            if isinstance(obj, torch.nn.Module):
                obj.to(device)
        self.device = torch.device(device)

    def process_sample_state(self, state: list):
        ''' process sample state
        '''
        state_ = [None] * len(state)
        for i in range(len(state)):
            state_[i] = torch.tensor(state[i], dtype=torch.float32, device=self.device).unsqueeze(0)
        return state_
     
    def create_model(self):
        ''' create model
        '''
        self.model = BaseNework(self.cfg, [None, 1])

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr) 

    def get_model_params(self):
        ''' get model params
        '''
        return self.model.state_dict()
    
    def put_model_params(self, model_params):
        ''' put model params
        '''
        self.model.load_state_dict(model_params)

    def get_action(self, state, **kwargs):
        ''' get action
        '''
        mode = kwargs.get('mode', None)
        if mode is None: 
            mode = 'sample' if self.cfg.mode == 'train' else 'predict'
        if mode == 'sample':
            return self.sample_action(state, **kwargs)
        elif mode == 'predict':
            return self.predict_action(state, **kwargs)
        else:
            raise NameError('[get_action] mode must be sample or predict')
        
    def sample_action(self, state, **kwargs):
        ''' sample action
        '''
        raise NotImplementedError
    def predict_action(self, state, **kwargs):
        ''' predict action
        '''
        raise NotImplementedError
    
    def update_policy_transition(self):
        ''' update policy transition
        '''
        self.policy_transition = {}
        
    def get_policy_transition(self):
        return self.policy_transition
    
    def create_summary(self):
        ''' create policy summary
        '''
        self.summary = {}
        self.summary['scalar'] = {}
        for i in range(len(self.action_size_list)):
            self.summary['scalar'][f'loss_{i}'] = 0.0

    def update_summary(self):
        ''' update policy summary
        '''
        for i in range(len(self.action_size_list)):
            self.summary['scalar'][f'loss_{i}'] = self.summary_loss[i]

    def get_summary(self):
        return self.summary['scalar']
    
    def learn(self, **kwargs):
        ''' learn policy
        '''
        self.prepare_data_before_learn(**kwargs)
    
    def prepare_data_before_learn(self, **kwargs):
        ''' prepare data before training
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        # multi-head state
        self.states = [ torch.tensor(states[i], dtype = torch.float32, device = self.device) for i in range(len(states)) ]
        self.next_states = [ torch.tensor(next_states[i], dtype = torch.float32, device = self.device) for i in range(len(next_states)) ]
        # multi-head action
        self.actions = [ torch.tensor(actions[i], dtype = torch.float32, device = self.device).unsqueeze(dim=1) for i in range(len(actions)) ]
        self.rewards = torch.tensor(rewards, dtype = torch.float32, device = self.device).unsqueeze(dim=1)
        self.dones = torch.tensor(dones, dtype = torch.float32, device = self.device).unsqueeze(dim=1)

    def update_data_after_learn(self):
        ''' update data after training
        '''
        self.data_after_train = {}
        
    def get_data_after_learn(self):

        return self.data_after_train
    
    def load_model_meta(self, model_meta):
        self.model_meta = model_meta
    
    def update_model_meta(self, dict):
        ''' update model meta
        '''
        self.model_meta.update(dict)

    def get_model_meta(self):
        return self.model_meta

    def save_model(self, fpath):
        ''' save model
        '''
        torch.save(self.model.state_dict(), fpath)

    def load_model(self, model_path = ''):
        ''' load model
        '''
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except:
            print(f"[BasePolicy.load_model] load model from {model_path} failed, please check the model path!")

class ToyPolicy:
    ''' base policy for traditional RL or non DRL
    '''
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.get_state_action_size()
        self.policy_transition = {}
        self.data_after_train = {}
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def get_state_action_size(self):
        self.n_states = self.obs_space.n
        self.n_actions = self.action_space.n
    def create_summary(self):
        ''' create policy summary
        '''
        self.summary = {
            'scalar': {
                'loss': 0.0,
            },
        }
    def update_summary(self):
        ''' update policy summary
        '''
        self.summary['scalar']['loss'] = self.loss
        
    def get_summary(self):
        return self.summary['scalar']
    
    def get_action(self, state, **kwargs):
        ''' get action
        '''
        mode = kwargs.get('mode', None)
        if mode is None: 
            mode = 'sample' if self.cfg.mode == 'train' else 'predict'
        if mode == 'sample':
            return self.sample_action(state, **kwargs)
        elif mode == 'predict':
            return self.predict_action(state, **kwargs)
        else:
            raise NameError('mode must be sample or predict')
        
    def sample_action(self, state, **kwargs):
        raise NotImplementedError
    def predict_action(self, state, **kwargs):
        raise NotImplementedError
    def update_policy_transition(self):
        ''' update policy transition
        '''
        self.policy_transition = {}
    def get_policy_transition(self):
        return self.policy_transition
    def update_data_after_learn(self):
        ''' update data after training
        '''
        self.data_after_train = {}
    def learn(self, **kwargs):
        ''' learn policy
        '''
        raise NotImplementedError
    def get_model_params(self):
        ''' get model parameters
        '''
        return copy.deepcopy(self.Q_table)
    
    def put_model_params(self, params):
        ''' set model parameters
        '''
        self.Q_table = params
    def save_model(self, fpath):
        ''' save model
        '''
        torch.save(obj=self.Q_table, f=fpath, pickle_module=dill)
    def load_model(self, fpath):
        ''' load model
        '''
        self.Q_table = None
        self.Q_table = torch.load(fpath, pickle_module=dill)