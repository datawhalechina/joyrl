#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 23:50:53
LastEditor: JiangJi
LastEditTime: 2023-05-27 20:51:28
Discription: 
'''

class Exp:
    def __init__(self, **kwargs) -> None:
        for k,v in kwargs.items():
            setattr(self,k,v)

            