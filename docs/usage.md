# Usage

This part mainly introduces the basic usage of `JoyRL`.

## Quick Start

### Config introduction

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

### Train and Test

To train an algorithm, we first need to change the `mode` to `train`, and configure the algorithm name `algo_name` and the environment name `env_name`, as well as the environment's `id`, and then set `max_episode` and `max_step`, as follows:

```yaml
general_cfg:
  algo_name: DQN 
  env_name: gym 
  device: cpu 
  mode: train 
  max_episode: -1 
  max_step: 200 
  load_checkpoint: false
  load_path: Train_single_CartPole-v1_DQN_20230515-211721
  load_model_step: best 
  seed: 1 
  online_eval: true 
  online_eval_episode: 10 
  model_save_fre: 500
env_cfg:
    id: CartPole-v1
    render_mode: null
```

Where `max_episode` indicates the maximum number of training rounds, setting it to -1 will continue training until stopped manually, `max_step` indicates the maximum number of steps per round, setting it to -1 will continue training until the environment returns `done=True` or `truncate=True`, **please set according to the actual environment situation**.

After configuring, you can start training by running any of the methods mentioned above, and during training, a `tasks` folder will be generated in the current directory, which contains the model files, log files, etc. generated during training, as follows:

<div align=center>
<img width="500" src="../figs/joyrl_docs/tasks_dir.png"/>
<div align=center>Figure 1 tasks folder composition</div>
</div>

Among them, the `logs` folder saves the log output by the terminal, the `models` folder saves the model files generated during training, the `tb_logs` folder saves the `tensorboard` files generated during training, such as reward curve, loss curve, etc., and the `results` folder saves the reward, loss, etc. in the form of `csv`, which is convenient for subsequent independent drawing and analysis. The `videos` folder saves the video files generated during the running process, which is mainly used in the test process. `config.yaml` saves the parameter configuration of the current running process, which is convenient for reproducing the training results.

If you want to test an algorithm, you just need to change the `mode` to `test`, and change the `load_checkpoint` (whether to load the model file) to `True`, and configure the model file path `load_path` and the model file step `load_model_step`, as follows:

```yaml
mode: test
load_checkpoint: true
load_path: Train_single_CartPole-v1_DQN_20230515-211721
load_model_step: 1000
```

### Online Evaluation Mode

During training, we often need to periodically test the policy in order to detect problems in time and save the best model. Therefore, `JoyRL` provides an online evaluation mode, which can be turned on by setting `online_eval` to `True`, and set `online_eval_episode` (the number of test rounds), as follows:

```yaml
online_eval: true
online_eval_episode: 10
model_save_fre: 500
```

Among them, `model_save_fre` indicates the model save frequency. When the online evaluation mode is turned on, each time the model is saved, an online evaluation will be performed, and an additional model named `best` will be output to save the model with the best test effect during the training process, but it is not necessarily the latest model.

### Multi-Process Mode

`JoyRL` supports multi-process mode, but unlike vectorized environment, `JoyRL`'s multi-process mode can run multiple interactors and learners asynchronously at the same time. The advantage of this is that if one interactor or learner fails, it will not affect the running of other interactors or learners, thus improving the stability of training. In `JoyRL`, the multi-process mode can be started by setting `n_interactors` and `n_learners` to an integer greater than 1, as follows:

```yaml
n_interactors: 2
n_learners: 2
```

Note that multi-learner mode is not supported yet, i.e. `n_learners` must be set to 1, and multi-learner mode will be supported in the future.

### Network Configuration

`JoyRL` supports building networks through configuration files, as follows:

```yaml

merge_layers:
  - layer_type: linear
    layer_size: [256]
    activation: relu
  - layer_type: linear
    layer_size: [256]
    activation: relu
```

This configuration is equivalent to:

```python

class MLP(nn.Module):
    def __init__(self, state_dim,action_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256) 
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, action_dim)  
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

Where the input `state_dim` and `action_dim` will be automatically obtained from the state action space of the environment, and the user only needs to configure the structure of the network.

### Multi-Head Network

In the previous section, we found that the input of network configuration is `merge_layers`, because `JoyRL` supports multi-head network, which means that multiple networks can be input at the same time, and then the outputs of multiple networks can be merged. For example, when the state input contains both image and linear input, we can configure two networks respectively, and then merge the outputs of the two networks. This is the use of multi-head network, as shown in the following figure:

<div align=center>
<img width="500" src="../figs/joyrl_docs/branch_merge.png"/>
<div align=center>Figure 2 branch and merge network</div>
</div>

Where `branch_layers` indicates the branch network, `merge_layers` indicates the merge network, the configuration method of `branch_layers` and `merge_layers` is the same, but you need to add `name` in the configuration of each network, as follows:

```yaml
branch_layers:
    - name: feature_1
      layers:
      - layer_type: conv2d
        in_channel: 4
        out_channel: 16 
        kernel_size: 4
        stride: 2
        activation: relu
      - layer_type: pooling
        pooling_type: max2d
        kernel_size: 2
        stride: 2
        padding: 0
      - layer_type: flatten
      - layer_type: norm
        norm_type: LayerNorm
        normalized_shape: 512
      - layer_type: linear
        layer_size: [128]
        activation: relu
    - name: feature_2
        layers:
        - layer_type: linear
          layer_size: [128]
          activation: relu
        - layer_type: linear
          layer_size: [128]
          activation: relu
merge_layers:
    - layer_type: linear
      layer_size: [256]
      activation: relu
    - layer_type: linear
      layer_size: [256]
      activation: relu
```

If it is a simple linear network, you can only configure `merge_layers` or `branch_layers`, and if it is a nonlinear network such as `CNN`, you can only configure `branch_layers`, because logically `merge_layers` can only receive linear input.

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
        super(CustomPolicy1, self).__init__(cfg)

class CustomPolicy2(DQNPolicy):
    ''' inherit DQNPolicy
    '''
    def __init__(self, cfg) -> None:
        super(CustomPolicy2, self).__init__(cfg)

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

