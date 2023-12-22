#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 13:55:56
LastEditor: JiangJi
LastEditTime: 2023-12-22 14:07:35
Discription: 
'''
import argparse
from joyrl import run

def main():
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--yaml', default=None, type=str,
                            help='the path of config file')
    args = parser.parse_args()
    run(yaml_path = args.yaml)
if __name__ == "__main__":
    main()