#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-01-26 09:59:44
Discription: 
'''
import copy
import torch
import torch.nn as nn
from joyrl.algos.base.base_layer import create_layer, LayerConfig
from joyrl.algos.base.action_layer import ActionLayerType, DiscreteActionLayer, ContinuousActionLayer, DPGActionLayer, DQNActionLayer
from joyrl.framework.config import MergedConfig
class BranchLayers(nn.Module):
    def __init__(self, branch_layers_cfg : list, input_size_list, **kwargs) -> None:
        super(BranchLayers, self).__init__(**kwargs)
        self.input_size_list = input_size_list
        self.branch_layers = nn.ModuleList([nn.ModuleList() for _ in range(len(self.input_size_list))])
        self.output_size_list = copy.deepcopy(input_size_list)
        self.branch_layers_cfg = branch_layers_cfg
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
    def __init__(self, merge_layers_cfg: list, input_size_list: list, **kwargs) -> None:
        super(MergeLayer, self).__init__(**kwargs)
        input_dim = sum([input_size[1] for input_size in input_size_list])
        self.input_size = [None, input_dim]
        self.output_size = copy.deepcopy(self.input_size)
        self.merge_layers = nn.ModuleList()
        self.merge_layers_cfg = merge_layers_cfg
        if self.merge_layers_cfg is None:
            self.merge_layers_cfg = []
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

class ActionLayers(nn.Module):
    def __init__(self, cfg: MergedConfig, input_size: list) -> None:
        super(ActionLayers, self).__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.action_size_list = self.cfg.action_size_list
        self.action_type_list = self.cfg.action_type_list
        assert len(self.action_type_list) == 1 and len(self.action_type_list) == len(self.action_size_list), "action_type_list and action_size_list must have the same length, and only one output"
        self.actor_outputs = []
        self.create_graph()

    def create_graph(self):
        self.action_layers = nn.ModuleList()
        for i in range(len(self.action_type_list)):
            action_type = ActionLayerType[self.action_type_list[i].upper()]
            action_size = self.action_size_list[i]
            if action_type == ActionLayerType.CONTINUOUS:
                action_layer = ContinuousActionLayer(self.cfg, self.input_size, action_size, id = i)
            elif action_type == ActionLayerType.DISCRETE:
                action_layer = DiscreteActionLayer(self.cfg, self.input_size, action_size,id = i)
            elif action_type == ActionLayerType.DPG:
                action_layer = DPGActionLayer(self.cfg, self.input_size, action_size, id = i)
            elif action_type == ActionLayerType.DQNACTION:
                action_layer = DQNActionLayer(self.cfg, self.input_size, action_size, id = i)
            else:
                raise ValueError("action_type must be specified in discrete, continuous or dpg")
            self.action_layers.append(action_layer)

    def forward(self, x, **kwargs):
        self.actor_outputs = []
        for i, action_layer in enumerate(self.action_layers):
            self.actor_outputs.append(action_layer(x, **kwargs))
        return self.actor_outputs
    
    def get_mus(self):
        mus = []
        for _, action_layer in enumerate(self.action_layers):
            mus.append(action_layer.get_mu())
        return mus[0]
    
    def get_qvalues(self):
        qvalues = []
        for _, action_layer in enumerate(self.action_layers):
            qvalues.append(action_layer.get_qvalue())
        return qvalues[0]
    
    def get_actions(self, **kwargs):
        actions = []
        for i, action_layer in enumerate(self.action_layers):
            actions.append(action_layer.get_action(**kwargs))
        return actions # [action_1, action_2, ...]
    
    def get_log_probs(self):
        return self.action_layers[0].get_log_prob()
    
    def get_log_probs_action(self, actions):
        return self.action_layers[0].get_log_prob_action(actions)
        # log_prob_sum = 0
        # for i, action_layer in enumerate(self.action_layers):
        #     log_prob_sum += action_layer.get_log_prob(actions[i])
        # return log_prob_sum
    
    def get_mean_entropy(self):
        return self.action_layers[0].get_mean_entropy()
        # entropy_sum = 0
        # for i, action_layer in enumerate(self.action_layers):
        #     entropy_sum += action_layer.get_entropy()
        # entropy_mean = entropy_sum / len(self.action_layers)
        # return entropy_mean
    
class BaseNework(nn.Module):
    def __init__(self, cfg: MergedConfig, input_size_list, **kwargs) -> None:
        super(BaseNework, self).__init__()
        self.cfg = cfg
        self.input_size_list = input_size_list
        
    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg.branch_layers, self.input_size_list)
        self.merge_layer = MergeLayer(self.cfg.merge_layers, self.branch_layers.output_size_list)

    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        return x
    
class QNetwork(BaseNework):
    ''' Q network, for value-based methods like DQN
    '''
    def __init__(self, cfg: MergedConfig, input_size_list: list) -> None:
        '''_summary_

        Args:
            cfg (_type_): _description_
            state_size (_type_): [[None, state_dim_1], [None, None, state_dim_2], ...]
            action_size (_type_): [action_dim_1, action_dim_2, ...]
        Raises:
            ValueError: _description_
        '''
        super(QNetwork, self).__init__(cfg, input_size_list)
        self.action_size_list = self.cfg.action_size_list
        self.dueling = hasattr(cfg, 'dueling') and cfg.dueling
        self.create_graph()

    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg.branch_layers, self.input_size_list)
        self.merge_layer = MergeLayer(self.cfg.merge_layers, self.branch_layers.output_size_list)
        action_type_list = ['dqnaction'] * len(self.cfg.action_type_list) 
        setattr(self.cfg, 'action_type_list', action_type_list)
        self.action_layers = ActionLayers(self.cfg, self.merge_layer.output_size)
        # if self.dueling:
        #     state_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
        #     self.state_value_layer, _ = create_layer(self.merge_layer.output_size, state_value_layer_cfg)
        #     action_value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_size_list[0]], activation='none')
        #     self.action_value_layer, _ = create_layer(self.merge_layer.output_size, action_value_layer_cfg)
        # else:
        #     action_layer_cfg = LayerConfig(layer_type='linear', layer_size=[self.action_size_list[0]], activation='none')
        #     self.action_value_layer, _ = create_layer(self.merge_layer.output_size, action_layer_cfg)

    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        actor_outputs = self.action_layers(x)
        return actor_outputs
    
    
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

   
class ActorCriticNetwork(BaseNework):
    ''' Value network, for policy-based methods,  in which the branch_layers and critic share the same network
    '''
    def __init__(self, cfg: MergedConfig) -> None:
        super(ActorCriticNetwork, self).__init__(cfg)
        self.action_type_list = self.cfg.action_type_list
        self.create_graph()
        
    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg.branch_layers, self.state_size_list)
        self.merge_layer = MergeLayer(self.cfg.merge_layers, self.branch_layers.output_size_list)
        self.value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
        self.value_layer, _ = create_layer(self.merge_layer.output_size, self.value_layer_cfg)
        self.action_layers = ActionLayers(self.cfg, self.merge_layer.output_size, self.action_type_list, self.action_size_list)
        
    def forward(self, x, pre_legal_actions=None):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        value = self.value_layer(x)
        action_outputs = self.action_layers(x, pre_legal_actions)
        return value, action_outputs


class ActorNetwork(BaseNework):
    def __init__(self, cfg: MergedConfig, input_size_list) -> None:
        super(ActorNetwork, self).__init__(cfg, input_size_list)
        self.action_type_list = self.cfg.action_type_list
        self.create_graph()
    
    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg.actor_branch_layers, self.input_size_list)
        self.merge_layer = MergeLayer(self.cfg.actor_merge_layers, self.branch_layers.output_size_list)
        self.action_layers = ActionLayers(self.cfg, self.merge_layer.output_size)
        
    def forward(self, x, pre_legal_actions=None):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        action_outputs = self.action_layers(x, pre_legal_actions = pre_legal_actions)
        return action_outputs

class CriticNetwork(BaseNework):
    def __init__(self, cfg: MergedConfig, input_size_list) -> None:
        super(CriticNetwork, self).__init__(cfg, input_size_list)
        self.create_graph()

    def create_graph(self):
        self.branch_layers = BranchLayers(self.cfg.critic_branch_layers, self.input_size_list)
        self.merge_layer = MergeLayer(self.cfg.critic_merge_layers, self.branch_layers.output_size_list)
        self.value_layer_cfg = LayerConfig(layer_type='linear', layer_size=[1], activation='none')
        self.value_layer, _ = create_layer(self.merge_layer.output_size, self.value_layer_cfg)
    
    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        value = self.value_layer(x)
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