[EN](./README.md)|中文

[中文](./README_cn.md)|EN
## JoyRL

## 安装说明

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
# GPU镜像安装
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
## 使用说明

直接更改 `config.config.GeneralConfig()`类中的参数比如环境名称(env_name)、算法名称(algo_name)等等，然后执行:
```bash
python main.py
```
运行之后会在目录下自动生成 `tasks`文件夹用于保存模型和结果。

或者也可以新建一个`yaml`文件自定义参数，例如 `config/custom_config_Train.yaml`然后执行:
```bash
python main.py --yaml config/custom_config_Train.yaml
```
在[defaults](./defaults/)文件夹中已经有一些预设的`yaml`文件，并且相应地在[benchmarks](./benchmarks/)文件夹中保存了一些已经训练好的结果。

## 算法列表

| 算法名称 |                          参考文献                           |                     作者                      | 备注 |
| :------: | :---------------------------------------------------------: | :-------------------------------------------: | :--: |
|   DQN    | [DQN Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) | [johnjim0816](https://github.com/johnjim0816) |      |
