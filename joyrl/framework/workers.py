#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-27 20:44:37
LastEditor: JiangJi
LastEditTime: 2023-05-27 20:44:41
Discription: 
'''
import ray
import numpy as np
@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, cfg, worker_id = 0 , env = None, logger = None):
        self.cfg = cfg
        self.worker_id = worker_id # Worker id
        self.worker_seed = self.cfg.seed + self.worker_id
        self.env = env
        self.logger = logger
    def run(self, data_server = None, learners = None, stats_recorder = None):
        ''' Run worker
        '''
        while not ray.get(data_server.check_episode_limit.remote()): # Check if episode limit is reached
            self.ep_reward, self.ep_step = 0, 0
            self.episode = ray.get(data_server.get_episode.remote())
            state, info = self.env.reset(seed = self.worker_seed)
            for _ in range(self.cfg.max_step):
                action = ray.get(learners[self.learner_id].get_action.remote(state, data_server=data_server)) # get action from learner
                next_state, reward, terminated, truncated , info = self.env.step(action) # interact with env
                self.ep_reward += reward
                self.ep_step += 1
                interact_transition = {'state':state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
                if self.cfg.share_buffer: # if all learners share the same buffer
                    ray.get(learners[0].add_transition.remote(interact_transition)) # add transition to learner
                    training_data = ray.get(learners[0].get_training_data.remote()) # get training data from learner
                else:
                    ray.get(learners[self.learner_id].add_transition.remote(interact_transition)) # add transition to data server
                    training_data = ray.get(learners[self.learner_id].get_training_data.remote()) # get training data from data server
                self.update_step, self.model_summary = ray.get(learners[self.learner_id].train.remote(training_data, data_server=data_server, logger = self.logger)) # train learner
                self.broadcast_model_params(learners) # broadcast model parameters to data server
                self.add_model_summary(stats_recorder) # add model summary to stats_recorder
                state = next_state # update state
                if terminated:
                    break
            self.logger.info.remote(f"Worker {self.worker_id} finished episode {self.episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
            ray.get(data_server.increase_episode.remote()) # increase episode count
            self.add_interact_summary(stats_recorder)  # add interact summary to stats_recorder
    def broadcast_model_params(self, learners = None):
        ''' Broadcast model parameters to data server
        '''
        #  aggregation model parameters
        # import torch
        # all_model_params = []
        # for learner in learners:
        #     all_model_params.append(ray.get(learner.get_model_params.remote()))
        # average_model_params = {}
        # for key in all_model_params[0].keys():
        #     average_model_params[key] = torch.mean(torch.stack([state_dict[key] for state_dict in all_model_params]), dim=0)
        # for learner in learners:
        #     ray.get(learner.set_model_params.remote(average_model_params))
        # broadcast model parameters
        # if self.learner_id == 0:
        if self.cfg.n_learners > 1:
            model_params = ray.get(learners[0].get_model_params.remote()) # 0 is the main learner
            for learner in learners[1:]:
                ray.get(learner.set_model_params.remote(model_params))
    def set_learner_id(self,learner_id):
        ''' Set learner id
        '''
        self.learner_id = learner_id

    def add_interact_summary(self,stats_recorder):
        ''' Add interact summary to stats_recorder
        '''
        summary = {
            'reward': self.ep_reward,
            'step': self.ep_step
        }
        ray.get(stats_recorder.add_interact_summary.remote((self.episode,summary)))

    def add_model_summary(self, stats_recorder):
        ''' Add model summary to stats_recorder
        '''
        if self.model_summary is not None:
            ray.get(stats_recorder.add_model_summary.remote((self.update_step,self.model_summary)))

class SimpleTester:
    ''' Simple online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        self.cfg = cfg
        self.env = env
        self.best_eval_reward = -float('inf')
    def eval(self,policy):
        ''' Evaluate policy
        '''
        sum_eval_reward = 0
        for _ in range(self.cfg.online_eval_episode):
            state, info = self.env.reset(seed = self.cfg.seed)
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            while True:
                action = policy.get_action(state, mode = 'predict')
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_step += 1
                if terminated or (0<= self.cfg.max_step <= ep_step):
                    sum_eval_reward += ep_reward
                    break
        mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
        if mean_eval_reward >= self.best_eval_reward:
            self.best_eval_reward = mean_eval_reward
            return True, mean_eval_reward
        return False, mean_eval_reward
@ray.remote    
class RayTester(SimpleTester):
    ''' Ray online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        super().__init__(cfg,env)
    