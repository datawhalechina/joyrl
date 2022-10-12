import sys,os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE" # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import gym
import torch
import datetime
import numpy as np
import argparse
import torch.nn as nn
from torch.distributions.categorical import Categorical

from common.utils import all_seed
from common.models import MLP
from common.launcher import Launcher
from envs.register import register_env
from ppo2 import PPO
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),np.array(self.actions),np.array(self.probs),\
                np.array(self.vals),np.array(self.rewards),np.array(self.dones),batches
                
    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
class Actor(nn.Module):
    def __init__(self,n_states, n_actions,
            hidden_dim=256):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
                nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, n_states,output_dim,hidden_dim=256):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, state):
        value = self.critic(state)
        return value
class Main(Launcher):
    def get_args(self):
        """ Hyperparameters
        """
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
        parser = argparse.ArgumentParser(description="hyperparameters")      
        parser.add_argument('--algo_name',default='PPO',type=str,help="name of algorithm")
        parser.add_argument('--env_name',default='CartPole-v0',type=str,help="name of environment")
        parser.add_argument('--continuous',default=False,type=bool,help="if PPO is continous") # PPO既可适用于连续动作空间，也可以适用于离散动作空间
        parser.add_argument('--train_eps',default=200,type=int,help="episodes of training")
        parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
        parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
        parser.add_argument('--batch_size',default=5,type=int) # mini-batch SGD中的批量大小
        parser.add_argument('--n_epochs',default=4,type=int)
        parser.add_argument('--actor_lr',default=0.0003,type=float,help="learning rate of actor net")
        parser.add_argument('--critic_lr',default=0.0003,type=float,help="learning rate of critic net")
        parser.add_argument('--gae_lambda',default=0.95,type=float)
        parser.add_argument('--policy_clip',default=0.2,type=float) # PPO-clip中的clip参数，一般是0.1~0.2左右
        parser.add_argument('--update_fre',default=20,type=int)
        parser.add_argument('--actor_hidden_dim',default=256,type=int)
        parser.add_argument('--critic_hidden_dim',default=256,type=int)
        parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
        parser.add_argument('--seed',default=10,type=int,help="seed") 
        parser.add_argument('--show_fig',default=False,type=bool,help="if show figure or not")  
        parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
        args = parser.parse_args()   
        default_args = {'result_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                        'model_path':f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
        }       
        args = {**vars(args),**default_args}  # type(dict)     
        return args
    def env_agent_config(self,cfg):
        ''' create env and agent
        '''
        register_env(cfg['env_name'])
        env = gym.make(cfg['env_name']) 
        if cfg['seed'] !=0: # set random seed
            all_seed(env,seed=cfg["seed"]) 
        try: # state dimension
            n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
        n_actions = env.action_space.n  # action dimension
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        models = {'Actor':Actor(cfg['n_states'],cfg['n_actions'], hidden_dim = cfg['actor_hidden_dim']),'Critic':Critic(cfg['n_states'],1,hidden_dim=cfg['critic_hidden_dim'])}
        memory =  PPOMemory(cfg["batch_size"]) # replay buffer
        agent = PPO(models,memory,cfg)  # create agent
        return env, agent
    def train(self,cfg,env,agent):
        ''' train agent
        '''
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        steps = 0
        for i_ep in range(cfg['train_eps']):
            state = env.reset()
            ep_reward = 0
            while True:
                action, prob, val = agent.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                steps += 1
                ep_reward += reward
                agent.memory.push(state, action, prob, val, reward, done)
                if steps % cfg['update_fre'] == 0:
                    agent.update()
                state = next_state
                if done:
                    break
            rewards.append(ep_reward)
            if (i_ep+1)%10==0:
                print(f"Episode: {i_ep+1}/{cfg['train_eps']}, Reward: {ep_reward:.2f}")
        print("Finish training!")
        return {'episodes':range(len(rewards)),'rewards':rewards}
    def test(self,cfg,env,agent):
        ''' test agent
        '''
        print("Start testing!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        for i_ep in range(cfg['test_eps']):
            state = env.reset()
            ep_reward = 0
            while True:
                action, prob, val = agent.predict_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(ep_reward)
            print(f"Episode: {i_ep+1}/{cfg['test_eps']}, Reward: {ep_reward:.2f}")
        print("Finish testing!")
        return {'episodes':range(len(rewards)),'rewards':rewards}

if __name__ == "__main__":
    main = Main()
    main.run()