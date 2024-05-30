#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-05-26 23:54:07
LastEditor: JiangJi
LastEditTime: 2024-05-28 10:03:20
Discription: 
'''
import ray
from threading import Lock

@ray.remote
class SharedData:
    ''' 
    '''
    def __init__(self, initial_value: float | int | dict) -> None:
        self.value = initial_value
        self.lock = Lock()

    def get_value(self):
        with self.lock:
            return self.value

    def set_value(self, new_value):
        with self.lock:
            self.value = new_value