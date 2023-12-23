# 多进程强化学习实现（以joyRL-DQN为例）

多进程可以充分地使用多核CPU的资源，强化学习中通过多进程可以在同一时刻让多个agent和environment并行交互产生样本，从而加速数据采样的速度。

## 1 多进程介绍

### 1.1 多进程的创建

1. 直接实例化线程

```python
from multiprocessing import Process
def function(n):  # 子进程
    pass

def run__process():  # 主进程
    process = [mp.Process(target=function, args=(i)) for i in range(5)] # 创建多进程
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束

if __name__ =='__main__':
    run__process()  
```

2. 通过类继承的方法来实现多进程（joyRL用的是该种方法）

```python
from multiprocessing import  Process

class MyProcess(Process): #继承Process类
    def __init__(self,name):
        super(MyProcess,self).__init__()
        self.name = name
 
    def run(self):
        print('测试%s多进程' % self.name)

if __name__ =='__main__':
    process = [MyProcess(i) for i in range(5)] # 创建多进程
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]   # 等待两个进程依次结束
```

### 1.2 进程间的交流

1. 共享内存

```python
import multiprocessing as mp
# 共享内存，多进程多核之间交流，必须用共享内存
value = mp.Value('d', 1)  # 第一个参数代表类型
array = mp.Array('i', [1, 2, 3])  # 第一个参数代表类型,第二个参数只能是一维的一个列表,不能是多维
```

2. Queue队列

```python
put()方法用以插入数据到队列中
get()方法可以从队列读取并且删除一个元素
```

## 2 多进程强化学习

### 2.1 多进程DQN算法流程图

多进程`DQN`中由一个`share_agent`和多个`local_agent_i`组成，其中：

- `share_agent`负责接收`local_agent_i`发出的梯度，并用该梯度更新自身网络，然后将更新好的参数再传递给`local_agent_i`

- `local_agent_i`（`i`用来区分不同智能体）负责和环境交互产生数据，计算梯度

下面展示DQN算法流程图

![mp_DQN](.\mp_DQN.svg)

### 2.2 具体步骤如下：

1. `local_agent`和环境交互产生数据

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/trainer.py 中97和98行）
action = self.local_agent.predict_action(state)
next_state, reward, terminated, truncated, info = self.env.step(action)
```

2. 计算`local_agent`中`policy_net`的损失函数

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 中119行）
loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)
```

3. 反向传播，得到`local_agent`中`policy_net`的梯度

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 中124行）
loss.backward()
```

4. 将`local_agent`中`policy_net`的梯度复制给`share_agent`中`policy_net`

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 中129和130行）
for param, share_param in zip(self.policy_net.parameters(), share_agent.policy_net.parameters()):
    share_param._grad = param.grad
```

5. 更新`share_agent`的`policy_net`网络参数

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 131行）
share_agent.optimizer.step()
```

6. 把`share_agent`的`policy_net`网络参数来复制给`local_agent`的`policy_net`网络参数

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 132行）
self.policy_net.load_state_dict(share_agent.policy_net.state_dict())
```

7. 固定间隔后用`local_agent`的`policy_net`网络参数更新`local_agent`的`target_net`网络参数

```python
# 对应的代码如下（在joyrl-offline/algos/DQN/agent.py 133和134行）
if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

