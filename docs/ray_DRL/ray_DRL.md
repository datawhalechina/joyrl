# 基于ray框架的分布式强化学习实现（以joyRL-DQN为例）

## 1 ray框架介绍

在Ray中，每个节点都运行着一个本地的Ray进程，每个进程之间通过网络进行通信。当我们提交一个任务到Ray集群时，Ray会将该任务的代码序列化并存储到Redis中，生成一个可执行的任务对象（Task Object）。每个任务对象都有一个唯一的任务ID，用于标识该任务。

ray官网： https://www.ray.io/

ray版本 2.3.0

### 基本概念：

在Ray中，任务（tasks）、Actor和对象（objects）是三个核心的概念，分别用于实现分布式计算、分布式状态管理和分布式对象存储。下面是它们的简单解释：

1. 任务（tasks）：任务是Ray中的基本计算单位，类似于函数或方法。任务可以被异步调用，也可以被提交到Actor中执行。Ray的任务调度器会将任务分配给可用的计算节点执行，这使得在分布式计算中能够方便地实现任务并行化和负载均衡。（简单理解就是函数）
2. Actor：Actor是一种可管理状态和行为的并发实体，用于实现分布式状态管理和共享状态。每个Actor实例都有一个唯一的标识符，称为Actor ID，可以使用该ID对Actor进行远程调用。Actor可以维护自己的状态，可以被异步调用，也可以被其他Actor或任务调用。Ray还提供了一种Actor之间消息传递的机制，使得Actor之间能够相互通信。（简单理解就是类）
3. 对象（objects）：对象是Ray中的分布式对象存储机制。对象可以被序列化和反序列化，并可以在计算节点之间传输。Ray还提供了一种对象引用的机制，使得可以在不同的计算节点之间共享对象。Ray中的对象是按需创建和销毁的，因此可以根据实际需求灵活地使用对象存储。（简单理解就是类或者函数的Id）

下面举个例子说明如何用ray进行分布式运算，和基本概念：

```
"""如果没有ray，第一次要pip install ray"""

#导入ray
import ray
# 开启ray
ray.init()

# 定义一个task任务
@ray.remote
def compute(num):
    return num * num
# 创建一个compute对象
results_objectID = [compute.remote(i) for i in range(10)]
print(f'results_objectID:{results_objectID}')
# 在多个节点上并行执行compute任务,ray.get获取结果。这里体现了分布式的思想，它是10个compute任务（函数）一起运行的
results = ray.get(results_objectID)
print(f'results:{results}')


# 定义一个Actor类
@ray.remote
class Counter(object):
    def __init__(self, init_value=0):
        self.value = init_value

    def increment(self):
        self.value += 1
        return self.value

result_list = []
# 创建一个Counter Actor对象
counter = Counter.remote()
#打印Counter Actor对象的objectRef
print(f'counter_objectID:{counter}')
# 在Counter Actor对象上调用increment方法
for i in range(10):
    result_list.append(ray.get(counter.increment.remote()))
print(f'result_list:{result_list}')
#关闭ray
ray.shutdown()
```


![results](.\results.png)


|    特性    |            multiprocessing             |                             Ray                              |
| :--------: | :------------------------------------: | :----------------------------------------------------------: |
|  设计理念  |          基于进程的并行计算库          |                    基于任务的分布式计算库                    |
|  编程模型  |       同步模型，使用共享内存和锁       |             异步模型，使用异步消息传递和事件驱动             |
| 分布式支持 | 主要面向单机多核计算，不原生支持分布式 | 具有强大的分布式支持，可在多个计算节点上执行任务，具有自动故障转移和任务调度功能 |
|  功能特性  |         提供基本的并行计算功能         | 提供一系列强化学习工具和算法，提供方便的任务调度和并发控制机制 |

## 2 ray-DQN

1. 整体思路

整体思路和multiprocessing一致，只不过是把共享变量写成Actor（类）了，主要包括ShareAgent，episode，best_reward，global_r_que。然后用了ray的语法。

修改的地方：

- 新增ray_run函数，为了用ray分布式训练RL（作用和multiprocessing中multi_run函数一致）

- 新增ShareAgent类，为了创建ShareAgent  （作用和multiprocessing中Agent类一致）

- Agent中加入了新的函数update_ray （作用和multiprocessing中update函数一致）

- 新增WorkerRay类，用于实现train和test的流程。（作用和multiprocessing中Worker类一致）

- 新增GlobalVarActor类，用于保存全局变量。（作用和multiprocessing中共享变量mp.Value('i', 0)， mp.Value('d', 0.)一致）

2. 关键代码

```
# 从share_agent获取share_policy_net
share_agent_policy_net = ray.get(self.share_agent.get_parameters.remote())
# 在用local_agent来更新share_policy_net
share_agent_policy_net = self.local_agent.update_ray(share_agent_policy_net)
# 将share_policy_net再传递给share_agent
ray.get(self.share_agent.receive_parameters.remote(share_agent_policy_net))
```





