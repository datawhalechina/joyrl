#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-08 13:37:26
Discription: 
'''
import copy
import torch
import torch.nn as nn
from joyrl.algos.base.base_layer import create_layer, LayerConfig
from joyrl.algos.base.action_layer import ActionLayerType, DiscreteActionLayer, ContinuousActionLayer, DPGActionLayer

class BranchLayers(nn.Module):
    def __init__(self, cfg, input_size_list, **kwargs) -> None:
        super(BranchLayers, self).__init__(**kwargs)
        self.input_size_list = input_size_list
        self.branch_layers = nn.ModuleList([nn.ModuleList() for _ in range(len(self.input_size_list))])
        self.output_size_list = copy.deepcopy(input_size_list)
        self.branch_layers_cfg = cfg.branch_layers
        if self.branch_layers_cfg is None:
            self.branch_layers_cfg = []
        for i, branch_layer_cfg in enumerate(self.branch_layers_cfg):
            if i >= len(input_size_list): # if branch_layers_cfg is more than input_size
                break
            input_size = input_size_list[i]
            layers_cfg_list = branch_layer_cfg['layers']
            for layer_cfg_dic in layers_cfg_list:
                if "layer_type" not in layer_cfg_dic:
                    raise ValueError("layer_type must be specified in layer_cfg")
                layer_cfg = LayerConfig(**layer_cfg_dic)
                layer, layer_output_size = create_layer(input_size, layer_cfg)
                self.branch_layers[i].append(layer)
                input_size = layer_output_size
            self.output_size_list[i] = layer_output_size
    def forward(self, x):
        if isinstance(x, torch.Tensor): # if x is a tensor, convert it to a list
            x = [x]
        for i, branch_layer in enumerate(self.branch_layers):
            for layer in branch_layer:
                x[i] = layer(x[i])
        return x
    def reset_noise(self):
        ''' reset noise for noisy layers
        '''
        for branch_layer in self.branch_layers:
            for layer in branch_layer:
                if hasattr(layer, "reset_noise"):
                    layer.reset_noise()
    
class MergeLayer(nn.Module):
    def __init__(self, cfg, input_size_list, **kwargs) -> None:
        super(MergeLayer, self).__init__(**kwargs)
        input_dim = sum([input_size[1] for input_size in input_size_list])
        self.input_size = [None, input_dim]
        self.output_size = copy.deepcopy(self.input_size)
        self.merge_layers = nn.ModuleList()
        self.merge_layers_cfg = cfg.merge_layers
        for layer_cfg_dic in self.merge_layers_cfg:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_output_size = create_layer(self.input_size, layer_cfg)
            self.merge_layers.append(layer)
            self.input_size = layer_output_size
            self.output_size = layer_output_size
    def forward(self, x):
        x = torch.cat(x, dim=1)
        for layer in self.merge_layers:
            x = layer(x)
        return x
    def reset_noise(self):
        ''' reset noise for noisy layers
        '''
        for layer in self.merge_layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()
    
class BaseNework(nn.Module):
    def __init__(self, cfg) -> None:
        super(BaseNework, self).__init__()
        self.cfg = cfg
        
    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg, [[None, 1], [None, 1]])
        self.merge_layer = MergeLayer(self.cfg, self.branch_layers.output_size_list)

    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        return x
    
class QNetwork(BaseNework):
    ''' Q network, for value-based methods like DQN
    '''
    def __init__(self, cfg, state_size_list, action_size_list):
        '''_summary_

        Args:
            cfg (_type_): _description_
            state_size (_type_): [[None, state_dim_1], [None, None, state_dim_2], ...]
            action_size (_type_): [action_dim_1, action_dim_2, ...]
        Raises:
            ValueError: _description_
        '''
        super(QNetwork, self).__init__(cfg)
        self.state_size_list = state_size_list
        self.action_size_list = action_size_list
        self.dueling = hasattr(cfg, 'dueling') and cfg.dueling
        self.create_graph()

    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg, self.state_size_list)
        self.merge_layer = MergeLayer(self.cfg, self.branch_layers.output_size_list)
        if self.dueling:
            state_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
            self.state_value_layer, layer_out_size = create_layer(self.merge_layer.output_size, state_value_layer_cfg)
            action_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_size_list[0]], activation='none')
            self.action_value_layer, layer_out_size = create_layer(self.merge_layer.output_size, action_value_layer_cfg)
        else:
            action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_size_list[0]], activation='none')
            self.action_value_layer, layer_out_size = create_layer(self.merge_layer.output_size, action_layer_cfg)

    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
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
        self.branch_layers.reset_noise()
        self.merge_layer.reset_noise()
        
class ValueNetwork(BaseNework):
    ''' Value network, for policy-based methods,  in which the branch_layers and critic share the same network
    '''
    def __init__(self, cfg, state_size, action_space) -> None:
        super(ValueNetwork, self).__init__(cfg)
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
    from joyrl.framework.config import MergedConfig
    import gymnasium as gym
    cfg = MergedConfig()
    state_size = [[None, 4], [None, 4]]
    cfg.n_actions = 2
    cfg.continuous = False
    cfg.min_policy = 0
    cfg.branch_layers = [

        [
            {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
            {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        ],
        # [
        #     {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        #     {'layer_type': 'linear', 'layer_size': [64], 'activation': 'ReLU'},
        # ],
    ]
    cfg.merge_layers = [
        {'layer_type': 'linear', 'layer_size': [2], 'activation': 'ReLU'},
        {'layer_type': 'linear', 'layer_size': [2], 'activation': 'ReLU'},
    ]
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
    model = QNetwork(cfg, state_size, [action_space.n])
    x = [torch.tensor([[ 0.0012,  0.0450, -0.0356,  0.0449]]), torch.tensor([[ 0.0012,  0.0450, -0.0356,  0.0449]])]
    x = model(x)
    print(x)
    # value_net = QNetwork(cfg, state_dim, cfg.n_actions)
    # print(value_net)
    # x = torch.tensor([36])
    # print(x.shape)
    # print(value_net(x))