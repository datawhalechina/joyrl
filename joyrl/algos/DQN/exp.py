#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 00:29:01
LastEditor: JiangJi
LastEditTime: 2023-04-16 00:29:01
Discription: 
'''

from joyrl.algos.base.exps import BaseExp

class Exp(BaseExp):
    def __init__(self, state=None, action=None, reward=None, next_state=None, terminated=None, **kwargs):
        super(Exp, self).__init__(state, action, reward, next_state, terminated)
        