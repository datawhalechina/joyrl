#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-25 09:28:26
LastEditor: JiangJi
LastEditTime: 2024-01-23 18:14:24
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
    def __init__(self):
        super(BaseActionLayer, self).__init__()
        pass

class DiscreteActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim, **kwargs):
        super(DiscreteActionLayer, self).__init__()
        self.cfg = cfg
        self.min_policy = cfg.min_policy
        if kwargs: self.id = kwargs['id']
        self.action_dim = action_dim
        output_size = input_size
        action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='leakyrelu')
        self.logits_p_layer, layer_out_size = create_layer(output_size, action_layer_cfg)

    def forward(self, x, pre_legal_actions = None):
        logits_p = self.logits_p_layer(x)
        if pre_legal_actions is not None:
            pre_legal_actions = logits_p.type(logits_p.dtype)
            large_negative = torch.finfo(torch.float16).min if logits_p.dtype == torch.float16 else -1e9
            mask_logits_p = logits_p * pre_legal_actions + (1 - pre_legal_actions) * large_negative
            probs = F.softmax(mask_logits_p, dim=1)
        else:
            probs = F.softmax(logits_p - logits_p.max(dim=1, keepdim=True).values, dim=1) # avoid overflow
            probs = (probs + self.min_policy) / (1.0 + self.min_policy * self.action_dim) # add a small probability to explore
        output = {"probs": probs}
        output.update(self.get_action(probs))
        return output
    
    def get_action(self, probs):
        ''' get action
        '''
        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action).detach()
        action = action.detach().cpu().numpy().item()
        return {"action": action, "log_probs": log_probs}

    def get_action(self, x, **kwargs):
        ''' get action
        '''
        pre_legal_actions = kwargs.get('pre_legal_actions', None)
        output = self.forward(x, pre_legal_actions)
        probs = output["probs"]
        dist = Categorical(probs)
        # action = torch.multinomial(probs, 1).squeeze(1)
        return action
        
class ContinuousActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim, **kwargs):
        super(ContinuousActionLayer, self).__init__()
        self.cfg = cfg
        self.min_policy = cfg.min_policy
        if kwargs: self.id = kwargs['id']
        self.action_dim = action_dim
        output_size = input_size
        mu_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='tanh')
        self.mu_layer, layer_out_size = create_layer(output_size, mu_layer_cfg)
        self.log_std = nn.Parameter(torch.zeros(1, self.action_dim))

    def forward(self,x):
        mu = self.mu_layer(x)
        sigma = torch.exp(self.log_std)
        # log_prob = -0.5 * (sigma.log() + ((mu - x) / sigma).pow(2) + math.log(2 * math.pi))
        # sigma = F.softplus(self.fc4(x)) + 0.001 # std of normal distribution, add a small value to avoid 0
        # sigma = torch.clamp(sigma, min=-0.25, max=0.25) # clamp the std between 0.001 and 1
        output = {"mu": mu, "sigma": sigma}
        return output
    def get_action(self, x, **kwargs):
        ''' get action
        '''
        output = self.forward(x)
        mu, sigma = output["mu"], output["sigma"]
        action = torch.normal(mu, sigma)
        return action
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
        