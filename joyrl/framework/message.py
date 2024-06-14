#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2024-02-25 15:46:04
LastEditor: JiangJi
LastEditTime: 2024-06-14 17:49:49
Discription: 
'''
from enum import Enum, unique
from typing import Optional, Any
from dataclasses import dataclass

@unique
class MsgType(Enum):
    # tracker
    TRACKER_GET_EPISODE = 0
    TRACKER_INCREASE_EPISODE = 1
    TRACKER_INCREASE_UPDATE_STEP = 2
    TRACKER_GET_UPDATE_STEP = 3
    TRACKER_CHECK_TASK_END = 4
    TRACKER_FORCE_TASK_END = 5

    # interactor
    INTERACTOR_SAMPLE = 10
    INTERACTOR_GET_SAMPLE_DATA = 11
    
    # learner
    LEARNER_UPDATE_POLICY = 20
    LEARNER_GET_UPDATED_MODEL_PARAMS_QUEUE = 21

    # collector
    COLLECTOR_PUT_EXPS = 30
    COLLECTOR_GET_TRAINING_DATA = 31
    COLLECTOR_GET_BUFFER_LENGTH = 32

    # recorder
    RECORDER_PUT_SUMMARY = 40
    
    # policy_mgr
    POLICY_MGR_PUT_MODEL_PARAMS = 70
    POLICY_MGR_PUT_MODEL_META = 71

@dataclass
class Msg(object):
    type: MsgType
    data: Optional[Any] = None