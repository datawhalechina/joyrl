#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-25 09:28:26
LastEditor: JiangJi
LastEditTime: 2024-07-21 16:46:03
Discription: 
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from enum import Enum
from joyrl.algos.base.base_layer import LayerConfig
from joyrl.algos.base.base_layer import create_layer


class ActionLayerType(Enum):
    ''' Action layer type
    '''
    DISCRETE = 1
    CONTINUOUS = 2
    DPG = 3
    DQNACTION = 4
    
class BaseActionLayer(nn.Module):
    def __init__(self,cfg, input_size, action_dim = 1, id = 0, **kwargs):
        super(BaseActionLayer, self).__init__()
        self.cfg = cfg
        self.id = id

    def get_action(self, **kwargs):
        mode = kwargs.get("mode", "sample")
        actor_output = kwargs.get("actor_output", None)
        if mode == "sample":
            return self.sample_action(**actor_output)
        elif mode == "predict":
            return self.predict_action(**actor_output)
        elif mode == "random":
            return self.random_action(**actor_output)
        else:
            raise NotImplementedError
        
class DQNActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_size, id = 0, **kwargs):
        super(DQNActionLayer, self).__init__(cfg=cfg, input_size=input_size, action_dim=action_size, id=id)
        self.action_dim = action_size[0]
        output_size = input_size
        self.dueling = hasattr(cfg, 'dueling') and cfg.dueling
        if self.dueling:
            state_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
            self.state_value_layer, _ = create_layer(output_size, state_value_layer_cfg)
            action_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='none')
            self.action_value_layer, _ = create_layer(output_size, action_value_layer_cfg)
        else:
            action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='none')
            self.action_value_layer, _ = create_layer(output_size, action_layer_cfg)

    def forward(self, x, **kwargs):
        if self.dueling:
            state_value = self.state_value_layer(x)
            action_value = self.action_value_layer(x)
            q_value = state_value + action_value - action_value.mean(dim=1, keepdim=True)
        else:
            q_value = self.action_value_layer(x)
        output = {"q_value": q_value}
        return output
    
    def sample_action(self, **kwargs):
        q_value = kwargs.get("q_value", None)
        return {"action": torch.argmax(q_value).detach().cpu().numpy().item()}
    
    def predict_action(self, **kwargs):
        q_value = kwargs.get("q_value", None)
        return {"action": torch.argmax(q_value).detach().cpu().numpy().item()}
    
    def random_action(self, **kwargs):
        return {"action": torch.randint(0, self.action_dim, (1,)).item()}

class DiscreteActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_size, id = 0, **kwargs):
        super(DiscreteActionLayer, self).__init__(cfg=cfg, input_size=input_size, action_dim=action_size, id=id, **kwargs)
        self.min_policy = cfg.min_policy
        self.action_dim = action_size[0]
        output_size = input_size
        action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='leakyrelu')
        self.logits_p_layer, _ = create_layer(output_size, action_layer_cfg)
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

    def sample_action(self, **kwargs):
        ''' get action
        '''
        probs = kwargs.get("probs", None)
        dist = Categorical(probs)
        action = dist.sample() # [batch_size]
        log_prob = dist.log_prob(action) # [batch_size]
        return {"action": action.detach().cpu().numpy().item(), "log_prob": log_prob.detach().cpu().numpy().item()}
    
    def predict_action(self,**kwargs):
        ''' get action
        '''
        probs = kwargs.get("probs", None)
        return {"action": torch.argmax(probs).detach().cpu().numpy().item(), "log_prob": None}
    
    def get_log_prob_action(self, actor_output, action):
        ''' get log_prob_action
        '''
        # action shape is [batch_size, action_dim]
        probs = actor_output.get("probs", None)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action.squeeze(dim=1))
        return log_prob.unsqueeze(dim=1)
    
    def get_entropy(self, actor_output):
        ''' get entropy
        '''
        probs = actor_output.get("probs", None)
        dist = Categorical(probs)
        entropy = dist.entropy()
        entropy_mean = torch.mean(entropy)
        return entropy_mean
 
class ContinuousActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim = 1, id = 0, **kwargs):
        super(ContinuousActionLayer, self).__init__(cfg=cfg, input_size=input_size, action_dim=action_dim, id = id, **kwargs)
        self.action_low = self.cfg.action_space_info.size[self.id][0]
        self.action_high = self.cfg.action_space_info.size[self.id][1]
        self.action_scale = (self.action_high - self.action_low)/2
        self.action_bias = (self.action_high + self.action_low)/2
        self.min_policy = cfg.min_policy
        output_size = input_size
        mu_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='tanh')
        self.mu_layer, _ = create_layer(output_size, mu_layer_cfg)
        self.action_std_scale = getattr(cfg, 'action_std_scale', 1.0)
        self.action_std_bias = getattr(cfg, 'action_std_bias', 0.0)
        self.log_std = nn.Parameter(torch.zeros(1, 1))

    def forward(self,x, **kwargs):
        mu = self.mu_layer(x) # [batch_size, 1]
        sigma = torch.ones_like(mu) * torch.exp(self.log_std)
        # log_prob = -0.5 * (sigma.log() + ((mu - x) / sigma).pow(2) + math.log(2 * math.pi))
        # sigma = F.softplus(self.fc4(x)) + 0.001 # std of normal distribution, add a small value to avoid 0
        # sigma = torch.clamp(sigma, min=-0.25, max=0.25) # clamp the std between 0.001 and 1
        mean = mu * self.action_scale + self.action_bias
        std = sigma * self.action_std_scale + self.action_std_bias
        output = {"mu": mu, "sigma": sigma, "mean": mean, "std": std}
        return output
    
    def sample_action(self, **kwargs):
        ''' get action
        '''
        mean = kwargs.get("mean", None)
        std = kwargs.get("std", None)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return {"action": action.detach().cpu().numpy().item(), "log_prob": log_prob}
    
    def predict_action(self, **kwargs):
        ''' get action
        '''
        mean = kwargs.get("mean", None)
        return {"action": mean.detach().cpu().numpy().item(), "log_prob": None}
    
    def random_action(self, **kwargs):
        return {"action": random.uniform(self.action_low, self.action_high)}
    
    def get_log_prob_action(self, actor_output, action):
        ''' get log_probs
        '''
        # action shape is [batch_size, action_dim]
        mean = actor_output.get("mean", None)
        std = actor_output.get("std", None)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)
        return log_prob
    
    def get_entropy(self, actor_output):
        ''' get entropy
        '''
        mean = actor_output.get("mean", None)
        std = actor_output.get("std", None)
        dist = Normal(mean, std)
        entropy = dist.entropy()
        entropy_mean = torch.mean(entropy)
        return entropy_mean
    
class DPGActionLayer(BaseActionLayer):
    def __init__(self, cfg, input_size, action_dim = 1, id = 0, **kwargs):
        super(DPGActionLayer, self).__init__(cfg=cfg, input_size=input_size, action_dim=action_dim, id=id)
        self.action_low = self.cfg.action_space_info.size[self.id][0]
        self.action_high = self.cfg.action_space_info.size[self.id][1]
        self.action_scale = (self.action_high - self.action_low)/2
        self.action_bias = (self.action_high + self.action_low)/2
        self.action_dim = action_dim
        self.output_size = input_size
        action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_dim], activation='tanh')
        self.action_layer, layer_out_size = create_layer(self.output_size, action_layer_cfg)
        self.output_size = layer_out_size
        
    def forward(self,x, **kwargs):
        mu = self.action_layer(x)
        mean = mu * self.action_scale + self.action_bias
        return {"mu": mu, "mean": mean}
        
    def sample_action(self, **kwargs):
        ''' get action
        '''
        return self.predict_action(**kwargs)
    
    def predict_action(self, **kwargs):
        ''' get action
        '''
        mu = kwargs.get("mu", None)
        action = mu * self.action_scale + self.action_bias
        return {"action": action.detach().cpu().numpy().item()}
    
    def random_action(self, **kwargs):
        return {"action": random.uniform(self.action_low, self.action_high)}
    
