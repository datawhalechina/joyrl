import copy, dill
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from gymnasium.spaces import Box, Discrete
from joyrl.algos.base.action_layer import ActionLayerType
class BasePolicy(nn.Module):
    ''' base policy for DRL
    '''
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device) 
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.optimizer = None
        self.policy_transition = {}
        self.data_after_train = {}
        self.get_state_size()
        self.get_action_size()

    def get_state_size(self):
        ''' get state size
        '''
        # state_size must be [[None, state_dim_1], [None, state_dim_2], ...]
        if isinstance(self.obs_space, Box):
            if len(self.obs_space.shape) == 3:
                self.state_size_list = [[None, self.obs_space.shape[0], self.obs_space.shape[1], self.obs_space.shape[2]]]
            else:
                self.state_size_list = [[None, self.obs_space.shape[0]]]
        elif isinstance(self.obs_space, Discrete):
            self.state_size_list = [[None, self.obs_space.n]]
        else:
            raise ValueError('obs_space type error')
        setattr(self.cfg, 'state_size_list', self.state_size_list)
        return self.state_size_list
    
    def get_action_size(self):
        ''' get action size
        '''
        # action_size must be [action_dim_1, action_dim_2, ...]
        if isinstance(self.action_space, Box):
            self.action_size_list = [self.action_space.shape[0]]
            self.action_type_list = ["CONTINUOUS"]
            self.action_high_list = [self.action_space.high[0]]
            self.action_low_list = [self.action_space.low[0]]  
        elif isinstance(self.action_space, Discrete):
            self.action_size_list = [self.action_space.n]
            self.action_type_list = ["DISCRETE"]
            self.action_high_list = [self.action_space.n]
            self.action_low_list = [0]
        else:
            raise ValueError('action_space type error')
        setattr(self.cfg, 'action_size_list', self.action_size_list)
        setattr(self.cfg, 'action_type_list', self.action_type_list)
        setattr(self.cfg, 'action_high_list', self.action_high_list)
        setattr(self.cfg, 'action_low_list', self.action_low_list)
    
    def create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr) 

    def get_model_params(self):
        ''' get model params
        '''
        return copy.deepcopy(self.state_dict())
    
    def put_model_params(self, model_params):
        ''' put model params
        '''
        self.load_state_dict(model_params)

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
        self.summary = {
            'scalar': {
                'loss': 0.0,
            },
        }
    def update_summary(self):
        ''' update policy summary
        '''
        self.summary['scalar']['loss'] = self.loss.item()
    def get_summary(self):
        return self.summary['scalar']
    def learn(self, **kwargs):
        ''' learn policy
        '''
        raise NotImplementedError
    def update_data_after_learn(self):
        ''' update data after training
        '''
        self.data_after_train = {}
    def get_data_after_learn(self):
        return self.data_after_train
    def save_model(self, fpath):
        ''' save model
        '''
        torch.save(self.state_dict(), fpath)
    def load_model(self, fpath):
        ''' load model
        '''
        try:
            self.load_state_dict(torch.load(fpath, map_location=self.device))
        except:
            print(f"load model from {fpath} failed, try to load model from cpu")

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