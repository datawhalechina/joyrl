
import argparse
import datetime, importlib, time, yaml
import os
import ray
import gymnasium as gym
from pathlib import Path
import torch.multiprocessing as mp
from joyrl.config.config import GeneralConfig, MergedConfig
from joyrl.framework.dataserver import DataServer
from joyrl.framework.stats import StatsRecorder, RayLogger, SimpleTrajCollector
from joyrl.framework.workers import Worker, RayTester   
from joyrl.framework.learners import Learner
from joyrl.utils.utils import merge_class_attrs, all_seed, save_cfgs

def load_cfgs(**kwargs):
    ''' load config from config class
    '''
    general_cfg = kwargs.get('general_cfg',None) # general config
    if general_cfg is None: general_cfg = GeneralConfig() # if not specified, use default config
    algo_cfg = kwargs.get('algo_cfg',None) # algorithm config
    if algo_cfg is None: algo_cfg = importlib.import_module(f"joyrl.algos.{general_cfg.algo_name}.config").AlgoConfig() 
    env_cfg = kwargs.get('env_cfg',None) # environment config
    if env_cfg is None: env_cfg = importlib.import_module(f"joyrl.envs.{general_cfg.env_name}.config").EnvConfig()
    return general_cfg,algo_cfg,env_cfg

def load_yaml_cfg(target_cfg, load_cfg, item):
    if load_cfg[item] is not None:
        for k, v in load_cfg[item].items():
            setattr(target_cfg, k, v)
    return target_cfg

def process_yaml_cfgs(general_cfg, algo_cfg, env_cfg):
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--yaml', default=None, type=str,

                        help='the path of config file')
    args = parser.parse_args()
    # load config from yaml file
    if args.yaml is not None:
        with open(args.yaml) as f:
            load_cfg = yaml.load(f, Loader=yaml.FullLoader)
            # load general config
            general_cfg = load_yaml_cfg(general_cfg, load_cfg, 'general_cfg')
            # load algo config
            algo_cfg = load_yaml_cfg(algo_cfg, load_cfg, 'algo_cfg')
            # load env config
            env_cfg = load_yaml_cfg(env_cfg, load_cfg, 'env_cfg')
    return general_cfg, algo_cfg, env_cfg

def merge_cfgs(general_cfg, algo_cfg, env_cfg):
    cfg = MergedConfig() # merge config
    cfg.general_cfg = general_cfg
    cfg.algo_cfg = algo_cfg
    cfg.env_cfg = env_cfg
    cfg = merge_class_attrs(cfg, general_cfg)
    cfg = merge_class_attrs(cfg, algo_cfg)
    cfg = merge_class_attrs(cfg, env_cfg)
    return cfg
def create_dirs(cfg):
    general_cfg = cfg.general_cfg
    env_cfg = cfg.env_cfg
    def config_dir(dir,name = None):
            Path(dir).mkdir(parents=True, exist_ok=True)
            setattr(cfg, name, dir)
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
    env_name = env_cfg.id if env_cfg.id is not None else general_cfg.env_name
    task_dir = f"{os.getcwd()}/tasks/{general_cfg.mode.capitalize()}_{env_name}_{general_cfg.algo_name}_{curr_time}"
    dirs_dic = {
        'task_dir':task_dir,
        'model_dir':f"{task_dir}/models",
        'res_dir':f"{task_dir}/results",
        'log_dir':f"{task_dir}/logs",
        'traj_dir':f"{task_dir}/traj",
        'video_dir':f"{task_dir}/videos",
        'tb_dir':f"{task_dir}/tb_logs"
    }
    for k,v in dirs_dic.items():
        config_dir(v,name=k)
    return cfg

def print_cfgs(cfg, logger):
    def print_cfg(cfg, name = ''):
        cfg_dict = vars(cfg)
        logger.info.remote(f"{name}:")
        logger.info.remote(''.join(['='] * 80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        logger.info.remote(tplt.format("Name", "Value", "Type"))
        for k, v in cfg_dict.items():
            if v.__class__.__name__ == 'list': # convert list to str
                v = str(v)
            if k in ['model_dir','tb_writter']:
                continue
            if v is None: # avoid NoneType
                v = 'None'
            if "support" in k: # avoid ndarray
                v = str(v[0])
            logger.info.remote(tplt.format(k, v, str(type(v))))
        logger.info.remote(''.join(['='] * 80))
    print_cfg(cfg.general_cfg,name = 'General Configs')
    print_cfg(cfg.algo_cfg,name = 'Algo Configs')
    print_cfg(cfg.env_cfg,name = 'Env Configs')

def check_resources(cfg):
    # check cpu resources
    if cfg.__dict__.get('n_workers',None) is None: # set n_workers to 1 if not set
        setattr(cfg, 'n_workers', 1)
    if not isinstance(cfg.n_workers,int) or cfg.n_workers<=0: # n_workers must >0
        raise ValueError("the parameter 'n_workers' must >0!")
    if cfg.n_workers > mp.cpu_count() - 1:
        raise ValueError("the parameter 'n_workers' must less than total numbers of cpus on your machine!")
    # check gpu resources
    if cfg.device == "cuda" and cfg.n_learners > 1:
        raise ValueError("the parameter 'n_learners' must be 1 when using gpu!")
    if cfg.device == "cuda":
        n_gpus_tester = 0.05
        n_gpus_learner = 0.9
    else:
        n_gpus_tester = 0
        n_gpus_learner = 0
    return n_gpus_tester, n_gpus_learner

def create_single_env(cfg):
    ''' create single env
    '''
    env_cfg = cfg.env_cfg
    env_cfg_dic = env_cfg.__dict__
    kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
    env = gym.make(**kwargs)
    if env_cfg.wrapper is not None:
        wrapper_class_path = env_cfg.wrapper.split('.')[:-1]
        wrapper_class_name = env_cfg.wrapper.split('.')[-1]
        env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
        env = getattr(env_wapper, wrapper_class_name)(env)
    return env

def envs_config(cfg,logger):
    ''' configure environment
    '''
    # register_env(env_cfg.id)
    envs = [] # numbers of envs, equal to cfg.n_workers
    for _ in range(cfg.n_workers):
        env = create_single_env(cfg)
        envs.append(env)
    setattr(cfg, 'obs_space', envs[0].observation_space)
    setattr(cfg, 'action_space', envs[0].action_space)
    logger.info.remote(f"obs_space: {envs[0].observation_space}, n_actions: {envs[0].action_space}")  # print info
    return envs

def policy_config(cfg):
    ''' configure policy and data_handler
    '''
    policy_mod = importlib.import_module(f"joyrl.algos.{cfg.algo_name}.policy")
        # create agent
    data_handler_mod = importlib.import_module(f"joyrl.algos.{cfg.algo_name}.data_handler")
    policy = policy_mod.Policy(cfg) 
    if cfg.load_checkpoint:
        policy.load_model(f"tasks/{cfg.load_path}/models/{cfg.load_model_step}")
    data_handler = data_handler_mod.DataHandler(cfg)
    return policy, data_handler

def run(**kwargs):
    general_cfg,algo_cfg,env_cfg = load_cfgs(**kwargs)
    general_cfg,algo_cfg,env_cfg = process_yaml_cfgs(general_cfg,algo_cfg,env_cfg)
    cfg = merge_cfgs(general_cfg, algo_cfg, env_cfg)
    cfg = create_dirs(cfg)
    ray.shutdown()
    ray.init(include_dashboard=True)
    logger = RayLogger.remote(cfg.log_dir) # create ray logger 
    print_cfgs(cfg,logger)
    n_gpus_tester, n_gpus_learner = check_resources(cfg)
    all_seed(cfg.seed) # set seed
    envs = envs_config(cfg,logger) # configure environment
    test_env = create_single_env(cfg) # create test environment
    online_tester = RayTester.options(num_gpus= n_gpus_tester).remote(cfg,test_env) # create online tester
    policy, data_handler = policy_config(cfg) # create policy and data_handler
    stats_recorder = StatsRecorder.remote(cfg) # create stats recorder
    data_server = DataServer.remote(cfg) # create data server
    learners = []
    for i in range(cfg.n_learners):
        learner = Learner.options(num_gpus= n_gpus_learner / cfg.n_learners).remote(cfg, learner_id = i, policy = policy,data_handler = data_handler, online_tester = online_tester)
        learners.append(learner)
    workers = []
    for i in range(cfg.n_workers):
        worker = Worker.remote(cfg, worker_id = i,env = envs[i], logger = logger)
        worker.set_learner_id.remote(i%cfg.n_learners)
        workers.append(worker)
    s_t = time.time()
    worker_tasks = [worker.run.remote(data_server = data_server,learners = learners,stats_recorder = stats_recorder) for worker in workers]
    ray.get(worker_tasks) # wait for all workers finish
    e_t = time.time()
    logger.info.remote(f"Finish {cfg.mode}ing! total time consumed: {e_t-s_t:.2f}s")
    save_cfgs( {'general_cfg': general_cfg, 'algo_cfg': algo_cfg, 'env_cfg': env_cfg}, cfg.task_dir)  # save config
    ray.shutdown() # shutdown ray