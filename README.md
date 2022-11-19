[中文](./README_cn.md)|EN
## JoyRL

## Install

```bash
conda create -n easyrl python=3.7
conda activate easyrl
pip install -r requirements
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
And there are presets yaml files in the [defaults](./defaults/) folder and well trained results in the [benchmarks](./benchmarks/) folder.

## Algorithms

|       Name       |                          Reference                           |                    Author                     | Notes |
| :--------------: | :----------------------------------------------------------: | :-------------------------------------------: | :---: |
| DQN | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |       |