The default parameter settings for the environment are stored in `joyrl/framework/envs/gym/config.py`, as follows:

```python
class GeneralConfig(object):
    ''' General parameters for running
    '''
    def __init__(self) -> None:
        # basic settings
        self.env_name = "gym" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.device = "cpu" # device to use
        self.seed = 0 # random seed
        self.max_episode = -1 # number of episodes for training, set -1 to keep running
        self.max_step = -1 # number of episodes for testing, set -1 means unlimited steps
        self.collect_traj = False # if collect trajectory or not
        # multiprocessing settings
        self.n_interactors = 1 # number of workers
        self.interactor_mode = "dummy" # dummy, only works when learner_mode is serial
        self.learner_mode = "serial" # serial, parallel, whether workers and learners are in parallel
        self.n_learners = 1 # number of learners if using multi-processing, default 1
        # online evaluation settings
        self.online_eval = False # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.model_save_fre = 500 # model save frequency per update step
        # load model settings
        self.load_checkpoint = False # if load checkpoint
        self.load_path = "Train_single_CartPole-v1_DQN_20230515-211721" # path to load model
        self.load_model_step = 'best' # load model at which step
        # stats recorder settings
        self.interact_summary_fre = 1 # record interact stats per episode
        self.policy_summary_fre = 100 # record update stats per update step
```

Note:

* `env_name`: name of environment, currently only supports `gym` environment, and will support custom environment later.
* `algo_name`: name of algorithm, such as `DQN`, `PPO`, etc., see [Algorithm Parameter Description](./algo_cfg.md) for details.
* `mode`: mode, `train` or `test`.
* `device`: device, `cpu` or `cuda`.
* `seed`: random seed, when `0`, no random seed is set.
* `max_episode`: maximum number of training episodes, when `-1`, no limit on the number of training rounds.
* `max_step`: maximum number of steps per episode, when `-1`, no limit on the maximum number of steps per episode, until the environment returns `done=True` or `truncate=True`, **please set according to the actual environment**.
* `collect_traj`: whether to collect trajectory, when `True`, collect trajectory, otherwise do not collect trajectory, generally used for imitation learning, inverse reinforcement learning, etc.
* `n_interactors`: number of interactors, default `1`, please set according to the actual situation.
* `interactor_mode`: interactor mode, valid when `n_interactors>1`, `dummy` or `ray`, default `dummy`, when `dummy`, the interactors are executed in series each time, when `ray`, the interactors are executed in parallel each time to collect samples.
* `learner_mode`: learner mode, `serial` or `parallel`, default `serial`, when `serial`, the interactors are executed in series each time, and then the learner is executed to update the policy, when `parallel`, the interactors and learners are executed in parallel to collect samples and update the policy respectively.
* `n_learners`: number of learners, default `1`, please set according to the actual situation.
* `online_eval`: whether to test online, when `True`, test online, otherwise do not test online. When online testing is turned on, an additional model named `best` will be output to save the best model during the training process, but it is not necessarily the latest model.
* `online_eval_episode`: number of online test episodes, please set according to the actual situation.
* `model_save_fre`: model file save frequency, be careful not to set it too small, otherwise it will affect the training efficiency.
* `load_checkpoint`: whether to load the model file, when `True`, load the model file, otherwise do not load the model file.
* `load_path`: model file path, valid when `load_checkpoint=True`.
* `load_model_step`: load the number of model files, `best` means to load the best model.
* `interact_summary_fre`: interactor statistics frequency, statistics the statistics of the interactor every several episodes, such as reward, etc., for complex tasks, it can be set to `10` to avoid, for simple tasks, it can be set to `1`.
* `policy_summary_fre`: learner statistics frequency, statistics the statistics of the learner every several update steps, such as loss, etc., be careful not to set it too small.
