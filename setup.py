#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 13:01:23
LastEditor: JiangJi
LastEditTime: 2024-06-17 14:43:29
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
        "gymnasium==0.29.1",
        "gymnasium[box2d]==0.29.1",
        "tensorboard==2.16.2",
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        "dill==0.3.8",
        "scipy==1.13.0",
        "pygame==2.5.2",
        "swig==4.2.1",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "six==1.16.0",
        "setuptools==69.5.1",
        "scipy==1.13.0",
        "PyYAML==6.0.1",
        "pydantic==1.10.15",
        "psutil==0.3.14",
        ""
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