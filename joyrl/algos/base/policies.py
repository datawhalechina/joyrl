import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
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
        self.get_state_action_size()
    def get_state_action_size(self):
        ''' get state and action size
        '''
        # state_size must be [[None, state_dim_1], [None, state_dim_2], ...]
        # action_size must be [action_dim_1, action_dim_2, ...]
        if isinstance(self.obs_space, Box):
            if len(self.obs_space.shape) == 3:
                self.state_size = [None, self.obs_space.shape[0], self.obs_space.shape[1], self.obs_space.shape[2]]
            else:
                self.state_size = [None, self.obs_space.shape[0]]
        elif isinstance(self.obs_space, Discrete):
            self.state_size = [None, self.obs_space.n]
        else:
            raise ValueError('obs_space type error')
        if isinstance(self.action_space, Box):
            self.action_size = [self.action_space.shape[0]]
        elif isinstance(self.action_space, Discrete):
            self.action_size = [self.action_space.n]
        else:
            raise ValueError('action_space type error')
        return self.state_size, self.action_size
    def create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr) 

    def get_model_params(self):
        ''' get model params
        '''
        return self.state_dict()
    
    def put_model_params(self, model_params):
        ''' put model params
        '''
        self.load_state_dict(model_params)

    def get_optimizer_params(self):
        return self.optimizer.state_dict()
    def set_optimizer_params(self, optim_params_dict):
        self.optimizer.load_state_dict(optim_params_dict)
    def get_action(self,state, mode = 'sample',**kwargs):
        ''' get action
        '''
        if mode == 'sample':
            return self.sample_action(state, **kwargs)
        elif mode == 'predict':
            return self.predict_action(state, **kwargs)
        else:
            raise NameError('mode must be sample or predict')
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
        self.load_state_dict(torch.load(fpath))

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
        self.summary['scalar']['loss'] = self.loss.item()
    def get_action(self, state, mode = 'sample', **kwargs):
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
    def save_model(self, fpath):
        raise NotImplementedError
    def load_model(self, fpath):
        raise NotImplementedError