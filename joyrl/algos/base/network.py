#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2024-06-12 00:06:25
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
        self.actor_outputs = {}
        for i, action_layer in enumerate(self.action_layers):
            self.actor_outputs[i] = action_layer(x, **kwargs)
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
        return qvalues
    
    def get_actions(self, **kwargs):
        mode = kwargs.get('mode', 'train')
        actor_outputs = kwargs.get('actor_outputs', None)
        actions = []
        for i, action_layer in enumerate(self.action_layers):
            action_layer_output = action_layer.get_action(mode = mode, actor_output = actor_outputs[i])
            actions.append(action_layer_output['action'])
        return actions
    
    def get_actions_and_log_probs(self, **kwargs):
        mode = kwargs.get('mode', 'train')
        actor_outputs = kwargs.get('actor_outputs')
        actions = []
        log_probs_sum = 0
        for i, action_layer in enumerate(self.action_layers):
            action_layer_output = action_layer.get_action(mode = mode, actor_output = actor_outputs[i])
            actions.append(action_layer_output['action'])
            log_probs_sum += action_layer_output['log_prob']
        return actions, log_probs_sum
    
    def get_log_probs_action(self, actor_outputs, actions):
        log_prob_sum = 0
        for i, action_layer in enumerate(self.action_layers):
            actor_output = actor_outputs[i]
            action = actions[i]
            log_prob = action_layer.get_log_prob_action(actor_output, action)
            log_prob_sum += log_prob
        return log_prob_sum
    
    def get_mean_entropy(self, actor_outputs):
        entropy_sum = 0
        for i, action_layer in enumerate(self.action_layers):
            entropy_sum += action_layer.get_entropy(actor_outputs[i])
        entropy_mean = entropy_sum / len(self.action_layers)
        return entropy_mean
     
class BaseNework(nn.Module):
    def __init__(self, cfg: MergedConfig, input_size_list) -> None:
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

    def forward(self, x):
        x = self.branch_layers(x)
        x = self.merge_layer(x)
        actor_outputs = self.action_layers(x)
        return {"actor_outputs": actor_outputs}
    
    def reset_noise(self):
        ''' reset noise for noisy layers
        '''
        self.branch_layers.reset_noise()
        self.merge_layer.reset_noise()


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
    
class ActorCriticNetwork(BaseNework):
    ''' Value network, for policy-based methods,  in which the branch_layers and critic share the same network
    '''
    def __init__(self, cfg: MergedConfig, input_size_list: list) -> None:
        super(ActorCriticNetwork, self).__init__(cfg, input_size_list)
        self.action_type_list = self.cfg.action_type_list
        self.create_graph()
        
    def create_graph(self):
        if getattr(self.cfg, 'independ_actor', False):
            self.actor = ActorNetwork(self.cfg, self.input_size_list)
            self.critic = CriticNetwork(self.cfg, self.input_size_list)
        else:
            self.branch_layers = BranchLayers(self.cfg.branch_layers, self.input_size_list)
            self.merge_layer = MergeLayer(self.cfg.merge_layers, self.branch_layers.output_size_list)
            self.value_layer, _ = create_layer(self.merge_layer.output_size, LayerConfig(layer_type='linear', layer_size=[1], activation='none'))
            self.action_layers = ActionLayers(self.cfg, self.merge_layer.output_size,)
        
    def forward(self, x, pre_legal_actions=None):
        if getattr(self.cfg, 'independ_actor', False):
            # since input x is a list, need to deepcopy it to avoid changing the original x
            actor_outputs = self.actor(copy.deepcopy(x), pre_legal_actions)
            value = self.critic(copy.deepcopy(x))
            return {'value': value, 'actor_outputs': actor_outputs}
        else:
            x = self.branch_layers(x)
            x = self.merge_layer(x)
            value = self.value_layer(x)
            actor_outputs = self.action_layers(x, pre_legal_actions = pre_legal_actions)
            return {'value': value, 'actor_outputs': actor_outputs}
        
    def get_actions_and_log_probs(self, **kwargs):
        if getattr(self.cfg, 'independ_actor', False):
            return self.actor.action_layers.get_actions_and_log_probs(**kwargs)
        else:
            return self.action_layers.get_actions_and_log_probs(**kwargs)
        
    def get_log_probs_action(self, actor_outputs, actions):
        if getattr(self.cfg, 'independ_actor', False):
            return self.actor.action_layers.get_log_probs_action(actor_outputs, actions)
        else:
            return self.action_layers.get_log_probs_action(actor_outputs, actions)
        
    def get_mean_entropy(self, actor_outputs):
        if getattr(self.cfg, 'independ_actor', False):
            return self.actor.action_layers.get_mean_entropy(actor_outputs)
        else:
            return self.action_layers.get_mean_entropy(actor_outputs)
    
    def get_actions(self, **kwargs):
        if getattr(self.cfg, 'independ_actor', False):
            return self.actor.action_layers.get_actions(**kwargs)
        else:
            return self.action_layers.get_actions(**kwargs)
        