#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-28 20:07:56
LastEditor: JiangJi
LastEditTime: 2023-05-28 20:08:00
Discription: 
'''
import joyrl
class GeneralConfig():
    ''' General parameters for running
    '''
    def __init__(self) -> None:
        # basic settings
        self.env_name = "gym" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.device = "cpu" # device to use
        self.seed = 1 # random seed
        self.max_episode = 100 # number of episodes for training, set -1 to keep running
        self.max_step = 200 # number of episodes for testing, set -1 means unlimited steps
        self.collect_traj = False # if collect trajectory or not
        self.n_workers = 1 # number of workers
        self.n_learners = 1 # number of learners if using multi-processing, default 1
        self.share_buffer = True # if all learners share the same buffer
        # online evaluation settings
        self.online_eval = True # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.model_save_fre = 500 # model save frequency per update step
        # load model settings
        self.load_checkpoint = False # if load checkpoint
        self.load_path = "Train_CartPole-v1_DQN_20230515-211721" # path to load model
        self.load_model_step = 'best' # load model at which step

class AlgoConfig():
    ''' algorithm parameters
    '''
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end to get fixed epsilon, i.e. epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay
        self.gamma = 0.95  # reward discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_type = 'REPLAY_QUE' # replay buffer type
        self.buffer_size = 100000  # replay buffer size
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        # value network layers config
        self.value_layers = [
            {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        ]
class EnvConfig:
    def __init__(self) -> None:
        super().__init__()
        self.id = "CartPole-v1" # 环境名称
        self.render_mode = None # render mode: None, rgb_array, human
        self.wrapper = None # 
        self.ignore_params = ["wrapper", "ignore_params"]
if __name__ == "__main__":
    general_cfg = GeneralConfig()
    algo_cfg = AlgoConfig()
    env_cfg = EnvConfig()
    joyrl.run(general_cfg = general_cfg, algo_cfg = algo_cfg, env_cfg = env_cfg)
    