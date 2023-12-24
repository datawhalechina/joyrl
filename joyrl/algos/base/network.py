#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2023-12-24 21:43:29
Discription: 
'''
import torch.nn as nn
from joyrl.algos.base.base_layer import create_layer, LayerConfig
from joyrl.algos.base.action_layers import ActionLayerType, DiscreteActionLayer, ContinuousActionLayer, DPGActionLayer

class BaseNework(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class QNetwork(BaseNework):
    ''' Q network, for value-based methods like DQN
    '''
    def __init__(self, cfg, state_size, action_size):
        super(QNetwork, self).__init__()
        self.cfg = cfg
        self.dueling = hasattr(cfg, 'dueling') and cfg.dueling
        self.layers_cfg_dic = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer)
        action_dim = action_size[0]
        if self.dueling:
            # state value
            state_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
            self.state_value_layer, layer_out_size = create_layer(output_size, state_value_layer_cfg)
            # action value
            action_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[action_dim], activation='none')
            self.action_value_layer, layer_out_size = create_layer(output_size, action_value_layer_cfg)
        else:
            action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[action_dim], activation='none')
            self.action_value_layer, layer_out_size = create_layer(output_size, action_layer_cfg)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.dueling:
            state_value = self.state_value_layer(x)
            action_value = self.action_value_layer(x)
            q_value = state_value + action_value - action_value.mean(dim=1, keepdim=True)
        else:
            q_value = self.action_value_layer(x)
        return q_value
    
    def reset_noise(self):
        ''' reset noise for noisy layers
        '''
        for layer in self.layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()
        
class ValueNetwork(BaseNework):
    ''' Value network, for policy-based methods,  in which the actor and critic share the same network
    '''
    def __init__(self, cfg, state_size, action_space) -> None:
        super(ValueNetwork, self).__init__()
        self.cfg = cfg
        self.continuous = action_space.continuous
        self.layers_cfg_dic = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
        self.value_layer, layer_out_size = create_layer(output_size, value_layer_cfg)
        if self.continuous:
            self.action_layer = ContinuousActionLayer(cfg, output_size, action_space)
        else:
            self.action_layer = DiscreteActionLayer(cfg, output_size, action_space)
    def forward(self, x, legal_actions=None):
        for layer in self.layers:
            x = layer(x)
        value = self.value_layer(x)
        if self.continuous:
            mu, sigma = self.action_layer(x)
            return value, mu, sigma
        else:
            probs = self.action_layer(x, legal_actions)
            return value, probs
        
class BaseActorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
class BaseCriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class ActorNetwork(BaseActorNetwork):
    def __init__(self, cfg, state_size, action_space) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_type = ActionLayerType[cfg.action_type.upper()]
        self.layers_cfg_dic = cfg.actor_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        if self.action_type == ActionLayerType.DISCRETE:
            self.action_layer = DiscreteActionLayer(cfg, output_size, action_space)
        elif self.action_type == ActionLayerType.CONTINUOUS:
            self.action_layer = ContinuousActionLayer(cfg, output_size, action_space)
        elif self.action_type == ActionLayerType.DPG:
            self.action_layer = DPGActionLayer(cfg, output_size, action_space)
        else:
            raise ValueError("action_type must be specified in discrete, continuous or dpg")
    def forward(self, x, legal_actions=None):
        for layer in self.layers:
            x = layer(x)
        if self.action_type == ActionLayerType.DISCRETE:
            probs = self.action_layer(x, legal_actions)
            return probs
        elif self.action_type == ActionLayerType.CONTINUOUS:
            output = self.action_layer(x)
            return output
        elif self.action_type == ActionLayerType.DPG:
            mu = self.action_layer(x)
            return mu

class CriticNetwork(BaseCriticNetwork):
    def __init__(self, cfg, input_size, output_dim = 1):
        super(CriticNetwork, self).__init__()
        self.cfg = cfg
        self.layers_cfg_dic = cfg.critic_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = input_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        head_layer_cfg = LayerConfig(layer_type='linear', layer_size=[output_dim], activation='none')
        self.head_layer, layer_out_size = create_layer(output_size, head_layer_cfg)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        value = self.head_layer(x)
        return value
        

if __name__ == "__main__":
    # test：export PYTHONPATH=./:$PYTHONPATH
    import torch
    from config.general_config import MergedConfig
    import gymnasium as gym
    cfg = MergedConfig()
    state_size = [None,4]
    cfg.n_actions = 2
    cfg.continuous = False
    cfg.min_policy = 0
    cfg.value_layers = [
        {'layer_type': 'embed', 'n_embeddings': 10, 'embedding_dim': 32, 'activation': 'none'},
        {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
        {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
    ]
    cfg.actor_layers = [
        {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
        {'layer_type': 'linear', 'layer_size': [256], 'activation': 'ReLU'},
    ]
    action_space = gym.spaces.Discrete(2)
    actor = ActorNetwork(cfg, state_size, action_space)
    x = torch.tensor([[ 0.0012,  0.0450, -0.0356,  0.0449]])
    x = actor(x)
    print(x)
    # value_net = QNetwork(cfg, state_dim, cfg.n_actions)
    # print(value_net)
    # x = torch.tensor([36])
    # print(x.shape)
    # print(value_net(x))