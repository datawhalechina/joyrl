#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-27 20:55:27
LastEditor: JiangJi
LastEditTime: 2024-01-13 18:45:54
Discription: 
'''
from joyrl.envs.register import register_env
class EnvConfig(object):
    def __init__(self) -> None:
        self.id = "CartPole-v1" # environment id
        register_env(self.id)
        self.render_mode = None # render mode: None, rgb_array, human
        self.wrappers = None # 
        self.ignore_params = ["wrappers", "ignore_params"]