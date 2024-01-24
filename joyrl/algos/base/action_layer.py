#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-25 09:28:26
LastEditor: JiangJi
LastEditTime: 2024-01-25 00:40:52
Discription: 
'''
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from joyrl.algos.base.base_layer import LayerConfig
from joyrl.algos.base.base_layer import create_layer

class ActionLayerType(Enum):
    ''' Action layer type
    '''
    DISCRETE = 1
    CONTINUOUS = 2
    DPG = 3
    
class BaseActionLayer(nn.Module):
    def __init__(self,cfg):
        super(BaseActionLayer, self).__init__(cfg)
        self.cfg = cfg

    def get_action(self, **kwargs):
        mode = kwargs.get("mode", "sample")
        if mode == "sample":
            return self.sample_action()
        elif mode == "predict":
            return self.predict_action()
        else:
            raise NotImplementedError

class DiscreteActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim,id = 0, **kwargs):
        super(DiscreteActionLayer, self).__init__(cfg=cfg)
        self.id = id
        self.min_policy = cfg.min_policy
        if kwargs: self.id = kwargs['id']
        self.action_dim = action_dim
        output_size = input_size
        action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='leakyrelu')
        self.logits_p_layer, layer_out_size = create_layer(output_size, action_layer_cfg)
        self.probs = None

    def forward(self, x, **kwargs):
        pre_legal_action = kwargs.get("pre_legal_action", None)   
        logits_p = self.logits_p_layer(x)
        if pre_legal_action is not None:
            pre_legal_action = logits_p.type(logits_p.dtype)
            large_negative = torch.finfo(torch.float16).min if logits_p.dtype == torch.float16 else -1e9
            mask_logits_p = logits_p * pre_legal_action + (1 - pre_legal_action) * large_negative
            probs = F.softmax(mask_logits_p, dim=1)
        else:
            probs = F.softmax(logits_p - logits_p.max(dim=1, keepdim=True).values, dim=1) # avoid overflow
            probs = (probs + self.min_policy) / (1.0 + self.min_policy * self.action_dim) # add a small probability to explore
        output = {"probs": probs}
        self.probs = probs
        return output

    def sample_action(self):
        ''' get action
        '''
        dist = Categorical(self.probs)
        action = dist.sample()
        action = action.detach().cpu().numpy().item()
        return action
    
    def predict_action(self):
        ''' get action
        '''
        return torch.argmax(self.probs).detach().cpu().numpy().item()
    
    def get_log_prob(self, action):
        ''' get log_probs
        '''
        # action shape is [batch_size]
        if isinstance(action, int):
            action = torch.tensor(action, dtype=torch.int64, device=self.probs.device)
        dist = Categorical(self.probs)
        log_prob = dist.log_prob(action)
        return log_prob
    
    def get_mean_entropy(self):
        ''' get entropy
        '''
        dist = Categorical(self.probs)
        entropy = dist.entropy()
        entropy_mean = torch.mean(entropy)
        return entropy_mean
 
class ContinuousActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim, id = 0, **kwargs):
        super(ContinuousActionLayer, self).__init__(cfg=cfg)
        self.id = id
        self.action_high = self.cfg.action_high_list[self.id]
        self.action_low = self.cfg.action_low_list[self.id]
        self.action_scale = torch.tensor((self.action_high - self.action_low)/2, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=0)
        self.action_bias = torch.tensor((self.action_high + self.action_high)/2, device=self.cfg.device, dtype=torch.float32).unsqueeze(dim=0)
        self.min_policy = cfg.min_policy
        if kwargs: self.id = kwargs['id']
        self.action_dim = action_dim
        output_size = input_size
        mu_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='tanh')
        self.mu_layer, layer_out_size = create_layer(output_size, mu_layer_cfg)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self,x, **kwargs):
        mu = self.mu_layer(x)
        sigma = torch.exp(self.log_std)
        # log_prob = -0.5 * (sigma.log() + ((mu - x) / sigma).pow(2) + math.log(2 * math.pi))
        # sigma = F.softplus(self.fc4(x)) + 0.001 # std of normal distribution, add a small value to avoid 0
        # sigma = torch.clamp(sigma, min=-0.25, max=0.25) # clamp the std between 0.001 and 1
        output = {"mu": mu, "sigma": sigma}
        self.mu = mu
        self.sigma = sigma
        self.mean = self.mu * self.action_scale + self.action_bias
        self.std = self.sigma
        return output
    
    def sample_action(self):
        ''' get action
        '''
        dist = Normal(self.mean,self.std)
        action = dist.sample()
        action = torch.clamp(action, torch.tensor(self.action_low, device=self.cfg.device, dtype=torch.float32), torch.tensor(self.action_high, device=self.cfg.device, dtype=torch.float32))
        return action.detach().cpu().numpy()[0]
    
    def predict_action(self):
        ''' get action
        '''
        return self.mean.detach().cpu().numpy()[0]
    
    def get_log_prob(self, action):
        ''' get log_probs
        '''
        # action shape is [batch_size, action_dim]
        dist = Normal(self.mean,self.std)
        log_prob = dist.log_prob(action)
        return log_prob
    
    def get_mean_entropy(self):
        ''' get entropy
        '''
        dist = Normal(self.mean,self.std)
        entropy = dist.entropy()
        entropy_mean = torch.mean(entropy)
        return entropy_mean
class DPGActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim, **kwargs):
        super(DPGActionLayer, self).__init__()
        self.cfg = cfg
        if kwargs: self.id = kwargs['id']
        self.action_dim = action_dim
        self.output_size = input_size
        action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='tanh')
        self.action_layer, layer_out_size = create_layer(self.output_size, action_layer_cfg)
        self.output_size = layer_out_size
    def forward(self,x):
        mu = self.action_layer(x)
        output = {"mu": mu}
        return output
        