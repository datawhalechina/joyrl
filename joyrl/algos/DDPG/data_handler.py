#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-07-21 14:45:35
Discription: 
'''
from joyrl.algos.base.data_handler import BaseDataHandler
import numpy as np
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
