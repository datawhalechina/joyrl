#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 13:42:56
LastEditor: JiangJi
LastEditTime: 2023-12-22 13:49:09
Discription: 
'''
import joyrl

if __name__ == "__main__":
    print(joyrl.__version__)
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path)