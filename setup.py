#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 13:01:23
LastEditor: JiangJi
LastEditTime: 2024-06-01 17:01:07
Discription: 
'''
import sys,os
from setuptools import setup, find_packages
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("joyrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

def get_install_requires() -> str:
    return [
        "ray[default]==2.6.3",
        "torch==2.2.1",
        "torchaudio==2.2.1",
        "torchvision==0.17.1",
        "gymnasium==0.29.1",
        "tensorboard==2.16.2",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "dill==0.3.8",
        "scipy==1.13.0",
        "pygame==2.5.2",
    ]

def get_extras_require() -> str:
    req = {
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    }
    return req

setup(
    name="joyrl",
    version=get_version(),
    description="A Library for Deep Reinforcement Learning",
    long_description=open(f"{curr_path}/README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/datawhalechina/joyrl",
    author="johnjim0816",
    author_email="johnjim0816@gmail.com",
    license="MIT",
    python_requires=">=3.7",
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    platforms = "any",
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    entry_points={
        "console_scripts": [
            "joyrl=joyrl.scripts.scripts:main",
        ],
    },
)