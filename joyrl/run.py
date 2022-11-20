from pathlib import Path
import datetime
import gym
from joyrl.common.utils import merge_class_attrs,all_seed,get_logger,save_results,save_cfgs,plot_rewards

class MergedConfig:
    def __init__(self) -> None:
        pass
def print_cfgs(cfg):
    ''' print parameters
    '''
    cfg_dict = vars(cfg)
    print(cfg.__dict__)
    print("Hyperparameters:")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in cfg_dict.items():
        if v.__class__.__name__ == 'list':
            v = str(v)
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))
def evaluate(cfg,trainer,env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _,eval_ep_reward,_ = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward/cfg.eval_eps
        return mean_eval_reward
def run(general_cfg,algo_cfg):
    cfgs = {'general_cfg':general_cfg,'algo_cfg':algo_cfg}
    cfg = MergedConfig() # merge config
    cfg = merge_class_attrs(cfg,cfgs['general_cfg'])
    cfg = merge_class_attrs(cfg,cfgs['algo_cfg'])
    print_cfgs(cfg) # print the configuration
    env = gym.make(cfg.env_name,new_step_api=True)
    all_seed(env,seed = cfg.seed) # set seed
    # create dirs
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
    task_dir = f"tasks/{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
    Path(task_dir).mkdir(parents=True, exist_ok=True)
    model_dir = f"{task_dir}/models/"
    res_dir = f"{task_dir}/results/"
    log_dir = f"{task_dir}/logs/"
    logger = get_logger(log_dir) # create the logger
    try: # state dimension
        n_states = env.observation_space.n # print(hasattr(env.observation_space, 'n'))
    except AttributeError:
        n_states = env.observation_space.shape[0] # print(hasattr(env.observation_space, 'shape'))
    try:
        n_actions = env.action_space.n  # action dimension
    except AttributeError:
        n_actions = env.action_space.shape[0]
        logger.info(f"action_bound: {abs(env.action_space.low.item())}") 
        setattr(cfg, 'action_bound', abs(env.action_space.low.item()))
    logger.info(f"n_states: {n_states}, n_actions: {n_actions}") # print info
    # update to cfg paramters
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)

    agent_mod = __import__(f"joyrl.algos.{cfg.algo_name}.agent", fromlist=['Agent'])
    agent = agent_mod.Agent(cfg) # create agent
    trainer_mod = __import__(f"joyrl.algos.{cfg.algo_name}.trainer", fromlist=['Trainer'])
    trainer = trainer_mod.Trainer() # create trainer
    if cfg.load_checkpoint:
        agent.load_model(f"tasks/{cfg.load_path}/models/")
    logger.info(f"Start {cfg.mode}ing!")
    logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
    rewards = []  # record rewards for all episodes
    steps = [] # record steps for all episodes
    if cfg.mode.lower() == 'train':
        best_ep_reward = -float('inf')
        for i_ep in range(cfg.train_eps):
            agent,ep_reward,ep_step = trainer.train_one_episode(env, agent, cfg)
            logger.info(f"Episode: {i_ep+1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
            rewards.append(ep_reward)
            steps.append(ep_step)
            # for _ in range
            if (i_ep+1)%cfg.eval_per_episode == 0:
                mean_eval_reward = evaluate(cfg, trainer,env, agent)
                if mean_eval_reward  >= best_ep_reward: # update best reward
                    logger.info(f"Current episode {i_ep+1} has the best eval reward: {mean_eval_reward:.3f}")
                    best_ep_reward = mean_eval_reward 
                    agent.save_model(model_dir) # save models with best reward
        # env.close()
    elif cfg.mode.lower() == 'test':
        for i_ep in range(cfg.test_eps):
            agent,ep_reward,ep_step = trainer.test_one_episode(env, agent, cfg)
            logger.info(f"Episode: {i_ep+1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
            rewards.append(ep_reward)
            steps.append(ep_step)
        agent.save_model(model_dir)  # save models
        # env.close()
    logger.info(f"Finish {cfg.mode}ing!")
    res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'steps':steps}
    save_results(res_dic, res_dir) # save results
    save_cfgs(cfgs, task_dir) # save config
    plot_rewards(rewards, title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}" ,fpath = res_dir)
