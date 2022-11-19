[EN](./README.md)|中文
# JoyRL

## 安装说明

```bash
conda create -n joyrl python=3.7
pip install joyrl
```
Torch:

```bash
# CPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU镜像安装
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## 使用说明

下述代码演示了如何使用`joyrl`，仅仅只需要设置参数(包括`GeneralConfig`和`AlgoConfig`两个类)，在[examples](./examples/)文件夹中也给出了一些示例, 并且在[benchmarks](./benchmarks/)文件夹中也给出了一些训好的结果.
```python
import joyrl as jj
import gym
class GeneralConfig():
    def __init__(self) -> None:
        self.env_name = "CartPole-v1" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train or test
        self.seed = 0 # random seed
        self.device = "cpu" # device to use
        self.train_eps = 100 # number of episodes for training
        self.test_eps = 20 # number of episodes for testing
        self.eval_eps = 10 # number of episodes for evaluation
        self.eval_per_episode = 5 # evaluation per episode
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = False
        self.load_path = "tasks" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not

class AlgoConfig():
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.95  # discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]
if __name__ == "__main__":
    general_cfg = GeneralConfig()
    algo_cfg = AlgoConfig()
    env = gym.make(general_cfg.env_name,new_step_api=True)
    jj.run.run(env,general_cfg,algo_cfg)
```

## 算法列表

| 算法名称 |                          参考文献                           |                     作者                      | 备注 |
| :------: | :---------------------------------------------------------: | :-------------------------------------------: | :--: |
|   DQN    | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |      |
