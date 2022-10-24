## JoyRL

## 4. 运行环境

主要依赖：Python 3.7、PyTorch 1.10.0、Gym 0.21.0。

### 4.1. 创建Conda环境
```bash
conda create -n easyrl python=3.7
conda activate easyrl # 激活环境
```
### 4.2. 安装Torch

安装CPU版本：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
```
安装CUDA版本：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
如果安装Torch需要镜像加速的话，点击[清华镜像链接](https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/)，选择对应的操作系统，如```win-64```，然后复制链接，执行：
```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
```
也可以使用PiP镜像安装（仅限CUDA版本）：
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
### 4.3. 检验CUDA版本Torch安装

CPU版本Torch请忽略此步，执行如下Python脚本，如果返回True说明CUDA版本安装成功:
```python
import torch
print(torch.cuda.is_available())
```
### 4.4. 安装Gym

```bash
pip install gym==0.21.0
```
如需安装Atari环境，则需另外安装

```bash
pip install gym[atari,accept-rom-license]==0.21.0
```

### 4.5. 安装其他依赖

项目根目录下执行：
```bash
pip install -r requirements.txt
```