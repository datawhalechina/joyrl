# JoyRL

[![PyPI](https://img.shields.io/pypi/v/joyrl)](https://pypi.org/project/joyrl/)  [![GitHub issues](https://img.shields.io/github/issues/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/issues) [![GitHub stars](https://img.shields.io/github/stars/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/stargazers) [![GitHub forks](https://img.shields.io/github/forks/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/network) [![GitHub license](https://img.shields.io/github/license/datawhalechina/joyrl)](https://github.com/datawhalechina/joyrl/blob/master/LICENSE)

`JoyRL` is a parallel reinforcement learning library based on PyTorch and Ray. Unlike existing RL libraries, `JoyRL` is helping users to release the burden of implementing algorithms with tough details, unfriendly APIs, and etc. JoyRL is designed for users to train and test RL algorithms with **only hyperparameters configuration**, which is mush easier for beginners to learn and use. Also, JoyRL supports plenties of state-of-art RL algorithms including **RLHF(core of ChatGPT)**(See algorithms below). JoyRL provides a **modularized framework** for users as well to customize their own algorithms and environments. 

## Install

⚠️ Note that donot install JoyRL through any mirror image!!!

```bash
# you need to install Anaconda first
conda create -n joyrl python=3.8
conda activate joyrl
pip install -U joyrl
```

Torch install:

Pip install is recommended, but if you encounter network error, you can try conda install or pip install with mirrors.

```bash
# pip CPU only
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
# if network error, then GPU with mirror image
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CPU only
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
# GPU 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

## Usage

### Quick Start

the following presents a demo to use joyrl. As you can see, first create a yaml file to **config hyperparameters**, then run the command as below in your terminal. That's all you need to do to train a DQN agent on CartPole-v1 environment.

```bash
joyrl --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```
or you can run the following code in your python file. 

```python
import joyrl
if __name__ == "__main__":
    print(joyrl.__version__)
    yaml_path = "./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml"
    joyrl.run(yaml_path = yaml_path)
```

### Offline Run

If you want to run from source code for debugging or other purposes, you can clone this repo:

```bash
git clone https://github.com/datawhalechina/joyrl.git
```

Then install the dependencies:

```bash
pip install -r requirements.txt
# if you have installed joyrl, you'd better uninstall it to avoid conflicts
pip uninstall joyrl
```

Then you can run the following command to train a DQN agent on CartPole-v1 environment.

```bash
python offline_run.py --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```

## Documentation

More tutorials and API documentation are hosted on [JoyRL docs](https://datawhalechina.github.io/joyrl/) or [JoyRL 中文文档](https://datawhalechina.github.io/joyrl-book/#/joyrl_docs/main).

## Algorithms

|       Name       |                          Reference                           |                    Author                     | Notes |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---: |
| Q-learning | [RL introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) | [johnjim0816](https://github.com/johnjim0816) |       |
| DQN | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) | |
| Double DQN  |     [DoubleDQN Paper](https://arxiv.org/abs/1509.06461)      | [johnjim0816](https://github.com/johnjim0816) | |
| Dueling DQN | [DuelingDQN Paper](https://arxiv.org/abs/1511.06581) | [johnjim0816](https://github.com/johnjim0816) | |
| NoisyDQN | [NoisyDQN Paper](https://arxiv.org/pdf/1706.10295.pdf) | [johnjim0816](https://github.com/johnjim0816) | |
| DDPG | [DDPG Paper](https://arxiv.org/abs/1509.02971) | [johnjim0816](https://github.com/johnjim0816) | |
| TD3 | [TD3 Paper](https://arxiv.org/pdf/1802.09477) | [johnjim0816](https://github.com/johnjim0816) | |