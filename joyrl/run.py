#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-12-22 13:16:59
LastEditor: JiangJi
LastEditTime: 2024-06-10 21:30:09
Discription: 
'''
import os,copy
import ray
import argparse,datetime,importlib,yaml
import gymnasium as gym
from pathlib import Path
from joyrl.framework.config import GeneralConfig, MergedConfig, DefaultConfig
from joyrl.framework.trainer import Trainer
from joyrl.framework.utils import merge_class_attrs, all_seed, create_module,exec_method

class Launcher(object):
    def __init__(self, **kwargs):
        self.custom_general_cfg = kwargs.get('general_cfg', None)
        self.custom_algo_cfg = kwargs.get('algo_cfg', None)
        self.custom_env_cfg = kwargs.get('env_cfg', None)
        self.custom_env = kwargs.get('env', None)
        self.custom_policy = kwargs.get('policy')
        self.custom_data_handler = kwargs.get('data_handler',None)
        self.custom_yaml_path = kwargs.get('yaml_path',None)
        self._get_default_cfg()  # get default config
        self._process_yaml_cfg()  # load yaml config
        self._merge_cfgs() # merge all configs
        self._check_other_cfgs() # check other configs
        self._config_dirs()  # create dirs
        self._save_cfgs({'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg, 'env_cfg': self.env_cfg})
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        
    def _get_default_cfg(self):
        ''' get default config
        '''
        self.general_cfg = GeneralConfig() # general config
        # load custom config
        if self.custom_general_cfg is not None:
            self.general_cfg = merge_class_attrs(self.general_cfg, self.custom_general_cfg)
        self.algo_cfg = importlib.import_module(f"joyrl.algos.{self.general_cfg.algo_name}.config").AlgoConfig()
        if self.custom_algo_cfg is not None:
            self.algo_cfg = merge_class_attrs(self.algo_cfg, self.custom_algo_cfg)
        self.env_cfg = importlib.import_module(f"joyrl.envs.{self.general_cfg.env_name}.config").EnvConfig()
        if self.custom_env_cfg is not None:
            self.env_cfg = merge_class_attrs(self.env_cfg, self.custom_env_cfg)

    def _process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml', default=None, type=str,
                            help='the path of config file')
        args = parser.parse_args()
        # load config from yaml file
        yaml_path = None
        if args.yaml is not None:
            yaml_path = args.yaml
        elif self.custom_yaml_path is not None:
            yaml_path = self.custom_yaml_path
        if yaml_path is not None:
            with open(yaml_path) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # load general config
                self._load_yaml_cfg(self.general_cfg,load_cfg,'general_cfg')
                # load algo config
                self.algo_cfg = importlib.import_module(f"joyrl.algos.{self.general_cfg.algo_name}.config").AlgoConfig()
                self._load_yaml_cfg(self.algo_cfg,load_cfg,'algo_cfg')
                # load env config
                self.env_cfg = importlib.import_module(f"joyrl.envs.{self.general_cfg.env_name}.config").EnvConfig()
                self._load_yaml_cfg(self.env_cfg, load_cfg, 'env_cfg')

    def _merge_cfgs(self):
        ''' merge all configs
        '''
        self.cfg = MergedConfig()
        setattr(self.cfg, 'general_cfg', self.general_cfg)
        setattr(self.cfg, 'algo_cfg', self.algo_cfg)
        setattr(self.cfg, 'env_cfg', self.env_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.general_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.algo_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.env_cfg)

    def _check_other_cfgs(self):
        buffer_type = getattr(self.cfg, 'buffer_type', None)
        if buffer_type is not None and buffer_type.lower().startswith('onpolicy'):
            setattr(self.cfg, 'on_policy', True)
        else:
            setattr(self.cfg, 'on_policy', False)
        
    def _save_cfgs(self, config_dict: dict):
        ''' save config
        '''
        with open(f"{self.cfg.task_dir}/config.yaml", 'w') as f:
            for cfg_type in config_dict:
                yaml.dump({cfg_type: config_dict[cfg_type].__dict__}, f, default_flow_style=False)

    def _load_yaml_cfg(self,target_cfg: DefaultConfig,load_cfg,item):
        if load_cfg[item] is not None:
            for k, v in load_cfg[item].items():
                setattr(target_cfg, k, v)

    def _config_dirs(self):
        def config_dir(dir,name = None):
            Path(dir).mkdir(parents=True, exist_ok=True)
            setattr(self.cfg, name, dir)
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        env_name = self.env_cfg.id if self.env_cfg.id is not None else self.general_cfg.env_name
        task_dir = f"{os.getcwd()}/tasks/{self.general_cfg.mode.capitalize()}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
        dirs_dic = {
            'task_dir':task_dir,
            'model_dir':f"{task_dir}/models",
            'res_dir':f"{task_dir}/results",
            'fig_dir':f"{task_dir}/figs",
            'log_dir':f"{task_dir}/logs",
            'traj_dir':f"{task_dir}/traj",
            'video_dir':f"{task_dir}/videos",
            'tb_dir':f"{task_dir}/tb_logs"
        }
        for k,v in dirs_dic.items():
            config_dir(v,name=k)

    def env_config(self):
        ''' create single env
        '''
        if self.custom_env is not None:
            env = self.custom_env
        else:
            env_cfg_dic = self.env_cfg.__dict__
            kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
            env = gym.make(**kwargs)
        setattr(self.cfg, 'obs_space', env.observation_space)
        setattr(self.cfg, 'action_space', env.action_space)
        if self.env_cfg.wrappers is None:
            return env
        for wrapper in self.env_cfg.wrappers: 
            wrapper_name = wrapper['wrapper_name']
            wrapper_kwargs = copy.deepcopy(wrapper)
            wrapper_kwargs.pop("wrapper_name")
            mm = importlib.import_module('joyrl.envs.gym')
            env_wapper_cls = eval(f'mm.wrappers.{wrapper_name}')
            try:
                env = env_wapper_cls(env, **wrapper_kwargs)
            except Exception as e:
                env = env_wapper_cls(env)
        
        setattr(self.cfg, 'obs_space', env.observation_space)
        setattr(self.cfg, 'action_space', env.action_space)
        return env
    
    def policy_config(self):
        ''' configure policy and data_handler
        '''
        policy_mod = importlib.import_module(f"joyrl.algos.{self.general_cfg.algo_name}.policy")
         # create agent
        data_handler_mod = importlib.import_module(f"joyrl.algos.{self.general_cfg.algo_name}.data_handler")
        policy = policy_mod.Policy(self.cfg) 
        if self.cfg.load_checkpoint:
            policy.load_model(f"tasks/{self.cfg.load_path}/models/{self.cfg.load_model_step}")
            policy.save_model(f"{self.cfg.model_dir}/{self.cfg.load_model_step}")
        data_handler = data_handler_mod.DataHandler(self.cfg)
        return policy, data_handler

    def run(self) -> None:
        ray.init()     
        env = self.env_config() # create single env
        policy, data_handler = self.policy_config() # configure policy and data_handler                         
        trainer = Trainer(self.cfg, name = "Trainer", env = env, policy = policy, data_handler = data_handler) # create trainer
        trainer.run()

def run(**kwargs):
    launcher = Launcher(**kwargs)
    launcher.run()

if __name__ == "__main__":
    launcher = Launcher()
    launcher.run()
