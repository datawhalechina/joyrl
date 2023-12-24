#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 23:02:13
LastEditor: JiangJi
LastEditTime: 2023-12-24 22:52:08
Discription: 
'''
import gymnasium as gym
from gymnasium.envs.registration import register

def register_env(env_name):
    if env_name == 'Racetrack-v0':
        register(
            id='Racetrack-v0',
            entry_point='envs.racetrack:RacetrackEnv',
            max_episode_steps=1000,
            kwargs={}
        )
    elif env_name == 'FrozenLakeNoSlippery-v1':
        register(
            id='FrozenLakeNoSlippery-v1',
            entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
            kwargs={'map_name':"4x4",'is_slippery':False},
        )
    elif env_name == 'CustomCliffWalking-v0':
        register(
            id='CustomCliffWalking-v0',
            entry_point='joyrl.envs.gym.toy_text.cliff_walking:CustomCliffWalkingEnv',
            max_episode_steps=1000,
            kwargs={}
        )
    else:
        pass

# if __name__ == "__main__":
#     import random
#     import gym
#     env = gym.make('FrozenLakeNoSlippery-v1')
#     num_steps = 1000000
#     state = env.reset()
#     n_actions = env.action_space.n
#     print(state)
#     for _ in range(num_steps) :
#         next_state, reward, done,_ = env.step(random.choice(range(n_actions)))
#         print(next_state)
#         if (done) :
#             _ = env.reset()
    