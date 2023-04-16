#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 23:25:28
LastEditor: JiangJi
LastEditTime: 2023-04-16 23:25:28
Discription: 存储一些基本的数据类型
'''
from enum import Enum

class BufferType(Enum):
    '''经验池类型'''
    ReplayBuffer = 1
    PER = 2