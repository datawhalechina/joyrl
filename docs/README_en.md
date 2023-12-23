[中文](./README.md)|EN

## JoyRL-Offline

## Install

Currently JoyRL support Python3.7 and Gym0.25.2

```bash
conda create -n joyrl python=3.7
conda activate joyrl
pip install -r requirements.txt
```
Torch:

```bash
# CPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# GPU with mirrors
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Install Mujoco

```bash
pip install mujoco==2.3.2
pip install mujoco-py==2.1.2.14
```
click [mujoco-releases](https://github.com/deepmind/mujoco/releases) and download mujoco 2.1.0, unzip and move to `C:/[username]/.mujoco/`, it should be like `C:/[username]/.mujoco/mujoco210`, then add `C:/[username]/.mujoco/mujoco210/bin` to SYSTEM PATH.

Note that if you are using Windows, you'd better to download [visual-cpp-build-tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/).

## Usage

you can simply change the parameters (like env_name, algo_name) in `config.config.GeneralConfig()` and run:
```bash
python main.py
```
then it will a new folder named `tasks` to save results and models.

Or you can custom parameters with a `yaml` file as you can seen in  `config/custom_config_Train.yaml` and run:
```bash
python main.py --yaml config/custom_config_Train.yaml
```
And there are presets yaml files in the [presets](./presets/) folder and well trained results in the [benchmarks](./benchmarks/) folder.

## Docs

please click [docs](./docs/README.md)

## Environments

Please click [envs](./envs/README.md) to read environments instruments.

## Algorithms
|                Name                |                                            Policy                                            |                                            Reference                                             |                        Author                        | Notes |
|:----------------------------------:|:------------------------------------------------------------------------------------------------:|:----------------------------------------------------:| :---: | :---: |
| [Monte Carlo](./algos/MonteCarlo/) |  | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) |    [johnjim0816](https://github.com/johnjim0816)     |  |
|   [Value Iteration](./algos/VI/)   |  | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) |       [guoshicheng](https://github.com/gsc579)       |  |
|  [Q-learning](./algos/QLearning/)  |  | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) |    [johnjim0816](https://github.com/johnjim0816)     |       |
|      [Sarsa](./algos/Sarsa/)       |  | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) |    [johnjim0816](https://github.com/johnjim0816)     |       |
|        [DQN](./algos/DQN/)         |                                       |                   [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)                    |    [johnjim0816](https://github.com/johnjim0816), [guoshicheng](https://github.com/gsc579) [(CNN)](./algos/DQN/)     |       |
|  [DoubleDQN](./algos/DoubleDQN/)   |                                               |                       [DoubleDQN Paper](https://arxiv.org/abs/1509.06461)                        |    [johnjim0816](https://github.com/johnjim0816)     |       |
|    [PER_DQN](./algos/PER_DQN/)     |                                                 |                        [PER_DQN Paper](https://arxiv.org/pdf/1511.05952)                         | [wangzhongren](https://github.com/wangzhongren-code) |       |
|   [NoisyDQN](./algos/NoisyDQN/)    |                                            |                      [NoisyDQN Paper](https://arxiv.org/pdf/1706.10295.pdf)                      | [wangzhongren](https://github.com/wangzhongren-code) |       |
|  [DRQN](./algos/DRQN/)   |                                               |                       [DRQN Paper](https://arxiv.org/abs/1507.06527)                        |    [johnjim0816](https://github.com/johnjim0816)     |  [understand LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)     |
| C51 |  | [C51 Paper](https://arxiv.org/abs/1707.06887) |  | also called Categorical DQN |
|  [REINFORCE](./algos/REINFORCE/)   |                            |              [REINFORCE Paper](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf)              |    [johnjim0816](https://github.com/johnjim0816)     |       |
|        [A2C](./algos/A2C/)         |        |    [A2C blog](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f)    |    [johnjim0816](https://github.com/johnjim0816)     |       |
| [A3C](./algos/A3C/) | On | [A3C paper](https://arxiv.org/pdf/1602.01783) | [johnjim0816](https://github.com/johnjim0816), [Ariel Chen](https://github.com/cr-bh) | |
|        [PPO](./algos/PPO/)         |                                                     |                          [PPO Paper](https://arxiv.org/abs/1707.06347)                           |    [johnjim0816](https://github.com/johnjim0816), [Wen Qiu](https://github.com/clorisqiu1)     |  PPO-clip, PPO-kl     |
|       [DDPG](./algos/DDPG/)        |                                                    |                          [DDPG Paper](https://arxiv.org/abs/1509.02971)                          |    [johnjim0816](https://github.com/johnjim0816)     |       |
|        [TD3](./algos/TD3/)         |                                                     |                          [TD3 Paper](https://arxiv.org/pdf/1802.09477)                           |    [johnjim0816](https://github.com/johnjim0816)     |       |
|      [SoftQ](./algos/SoftQ/)       |                                                   |                         [SoftQ Paper](https://arxiv.org/abs/1702.08165)                          |    [johnjim0816](https://github.com/johnjim0816)     |       |
|       [GAIL](./algos/GAIL/)        |                                                    |                          [GAIL Paper](https://arxiv.org/abs/1606.03476)                          |      [Yi Zhang](https://github.com/ai4drug)      |       |
