# Usage

This part mainly introduces the basic usage of `JoyRL`.

## Quick Start

`JoyRL` aims to let users practice related reinforcement learning only by adjusting parameters, including:

* General parameters (`GeneralConfig`): parameters related to the running mode, such as the algorithm name `algo_name`, the environment name `env_name`, the random seed `seed`, etc .;
* Algorithm parameters (`AlgoConfig`): parameters related to the algorithm itself, which are also the main parameters that users need to adjust;
* Environment parameters (`EnvConfig`): environment-related parameters, such as `render_mode` in the `gym` environment;

`JoyRL` provides a variety of ways to configure hyperparameters, including `yaml` files, `python` files, etc., among which `yaml` files are the recommended configuration method for novices. Taking `DQN` as an example, users can create a new `yaml` file, for example `DQN.yaml`, and configure the relevant parameters in it, and execute:

```python
import joyrl

if __name__ == "__main__":
    print(joyrl.__version__) # print version
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path)
```

to start training. For the configuration of `yaml` files, `JoyRL` provides built-in terminal commands to execute, that is:

```bash
joyrl --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```

Users can also create a `python` file to customize the relevant parameter classes to run, as follows:

```python
import joyrl

class GeneralConfig:
    ''' General parameters for running
    '''
    def __init__(self) -> None:
        # basic settings
        self.env_name = "gym" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.interactor_mode = "dummy" # dummy, only works when learner_mode is serial
        self.learner_mode = "serial" # serial, parallel, whether workers and learners are in parallel
        self.device = "cpu" # device to use
        self.seed = 0 # random seed
        self.max_episode = -1 # number of episodes for training, set -1 to keep running
        self.max_step = 200 # number of episodes for testing, set -1 means unlimited steps
        self.collect_traj = False # if collect trajectory or not
        # multiprocessing settings
        self.n_interactors = 1 # number of workers
        # online evaluation settings
        self.online_eval = True # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.model_save_fre = 500 # model save frequency per update step
        # load model settings
        self.load_checkpoint = False # if load checkpoint
        self.load_path = "Train_single_CartPole-v1_DQN_20230515-211721" # path to load model
        self.load_model_step = 'best' # load model at which step

class EnvConfig(object):
    def __init__(self) -> None:
        self.id = "CartPole-v1" # environment id

if __name__ == "__main__":
    general_cfg = GeneralConfig()
    env_cfg = EnvConfig()
    joyrl.run(general_cfg = general_cfg, env_cfg = env_cfg)
```

Note that you must pass in the `kwarg` form to the `joyrl.run` function accurately!!!

At the same time, `JoyRL` comes with default parameter configuration. When the user passes in custom parameters, the parameters in the `yaml` file will be considered first, followed by the parameters passed in, and the default parameter configuration has the lowest priority. Users do not need to configure all parameters at the same time, and use the default configuration for some parameters that they don't care about.

## Custom Policy

Users can customize the policy by inheriting any policy class in `algos`, as follows:

```python
import joyrl
from joyrl.algos.base import BasePolicy
from joyrl.algos.DQN.policy import Policy as DQNPolicy

class CustomPolicy1(BasePolicy):
    ''' inherit BasePolicy
    '''
    def __init__(self, cfg) -> None:
        super(BasePolicy, self).__init__(cfg)

class CustomPolicy2(DQNPolicy):
    ''' inherit DQNPolicy
    '''
    def __init__(self, cfg) -> None:
        super(DQNPolicy, self).__init__(cfg)

if __name__ == "__main__":
    my_policy = CustomPolicy1()
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path,policy = my_policy)
```

Note that the custom policy must pass in the `cfg` parameter to facilitate the `JoyRL` framework to import the corresponding parameter configuration.

## Custom Environment

`JoyRL` also supports custom environments, as follows:

```python
class CustomEnv:
    def __init__(self,*args,**kwargs):
        pass
    def reset(self, seed = 0):
        return state, info
    def step(self, action):
        return state, reward, terminated, truncated, info
if __name__ == "__main__":
    my_env = CustomEnv()
    yaml_path = "xxx.yaml"
    joyrl.run(yaml_path = yaml_path, env = my_env)
```

Note that only the `gymnasium` interface environment is currently supported, that is, it must contain functions such as `reset` and `step`.

