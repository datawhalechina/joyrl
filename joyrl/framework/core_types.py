#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-06-16 08:53:47
LastEditor: JiangJi
LastEditTime: 2024-07-20 13:08:16
Discription: 
'''
from enum import Enum
from typing import Union

class ActionType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2
    DPG = 3
    DQNACTION = 4

class ActionSpaceInfo:
    ''' 
        i.e:
            CartPole-v1: type=[ActionType.DISCRETE], size=[2] or size=[[2]]
            Pendulum-v1: type=[ActionType.CONTINUOUS], size=[[-2,2]]
            Some: type=[ActionType.DISCRETE, ActionType.CONTINUOUS], size = [2, [-2,2]] or [[2], [-2,2]]
    '''
    def __init__(self,
                type: list[ActionType],
                size: list[Union[list[int], list[float], int]]
                ): 
        self.type = type
        self.size = size
        self._check_type_size()
        
    def _check_type_size(self):
        assert len(self.type) == len(self.size), 'action type and size must have the same length'
        for i in range(len(self.type)):
            if self.type[i] == ActionType.DISCRETE:
                if isinstance(self.size[i], list):
                    assert len(self.size[i]) == 1 and isinstance(self.size[i][0], int), f"action head {i} size expected to be a list with one int element, but got {self.size[i]}"
                elif isinstance(self.size[i], int):
                    self.size[i] = [self.size[i]]
                else:
                    raise ValueError(f"action head {i} size error, expected to be a list or int, but got {self.size[i]}")
            elif self.type[i] == ActionType.CONTINUOUS:
                if isinstance(self.size[i], list):
                    assert len(self.size[i]) == 2, f"action head {i} size expected to be 2, but got {len(self.size[i])}"
                    self.size[i] = [float(x) for x in self.size[i]]
                    assert self.size[i][0] < self.size[i][1], f"action head {i} size error, low must be less than high, but got {self.size[i]}"
                else:
                    raise ValueError(f"action head {i} size error, expected to be a list like [low,high], but got {self.size[i]}")
            elif self.type[i] == ActionType.DPG:
                if isinstance(self.size[i], list):
                    assert len(self.size[i]) == 2, f"action head {i} size expected to be 2, but got {len(self.size[i])}"
                    self.size[i] = [float(x) for x in self.size[i]]
                    assert self.size[i][0] < self.size[i][1], f"action head {i} size error, low must be less than high, but got {self.size[i]}"
            else:
                raise ValueError('action type error')

class ObsType(Enum):
    VECTOR = 1
    IMAGE = 2

class ObsSpaceInfo:
    ''' 
        i.e:
            CartPole-v1: size=[[4]]
            Pendulum-v1: size=[[3]]
            CarRacing-v2: size=[[3,96,96]]
            Some: size=[[4],[3],[3,96,96]]
    '''
    def __init__(self,
                type: list[ObsType],
                size: list[list[int]]
                ):
        self.type = type
        self.size = size
        self._check_type_size()
    
    def _check_type_size(self):
        assert len(self.type) == len(self.size), 'obs type and size must have the same length'
        for i in range(len(self.type)):
            if self.type[i] == ObsType.VECTOR:
                assert isinstance(self.size[i], list) and len(self.size[i]) == 1 and isinstance(self.size[i][0], int), f"obs head {i} size expected to be a list with one int element, but got {self.size[i]}"
            elif self.type[i] == ObsType.IMAGE:
                assert isinstance(self.size[i], list) and len(self.size[i]) == 3 and all(isinstance(x, int) for x in self.size[i]), f"obs head {i} size expected to be a int list like [C,H,W], but got {self.size[i]}"    
            else:
                raise ValueError('obs type error')
            self.size[i] = [None] + self.size[i]