## 多进程为什么有时比单进程还慢？

主要原因有：

* 进程之间通信消耗，包括本身的基础消耗和通信量的消耗
* 使用了很多锁，导致进程阻塞
* 使用了大量的`fork`，导致内存消耗过大，进而导致内存交换，进而导致进程阻塞
* CPU资源不足（对开发者无参考意义）

解决方法：
* 尽量减少进程间的通信，使用共享内存，保证每个进程执行时间足够长
  * 通信基础消耗：无论是`Ray`还是`multiprocessing`，都需要将数据序列化，然后再反序列化，这个过程是比较耗时的，此时如果进程执行时间太短，通信开销会比较大，后面会举例说明；开发`RL`框架时，对于`Ray`，可以将`Actor`的`__init__`和`__call__`函数尽量简单，将任务执行的逻辑放在`run`函数中，这样可以减少通信开销；对于`multiprocessing`，可以使用`Manager`来共享数据，而不是使用`Queue`，`Queue`的通信开销比较大。
  

通信消耗示例，程序如下：

```python
import ray
import time
import multiprocessing as mp

def exec_timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间：{execution_time} 秒")
        return result
    return wrapper

class BaseWorker:
    def __init__(self,id = 0) -> None:
        self.id = id
    def run(self):
        # 模拟任务执行
        time.sleep(0.0001)
        # time.sleep(1)
        return self.id
    
@ray.remote
class RayWorker(BaseWorker):
    def __init__(self,id = 0) -> None:
        super().__init__(id)

@exec_timer
def train_single(cfg):
    workers = [BaseWorker(id=i) for i in range(cfg.n_interactors)]
    for i in range(cfg.n_interactors):
        workers[i].run()

@exec_timer
def train_mp(cfg):
    processes = []
    workers = [BaseWorker(id=i) for i in range(cfg.n_interactors)]
    for i in range(cfg.n_interactors):
        processes.append(mp.Process(target=workers[i].run))
        processes[i].start()
    for i in range(cfg.n_interactors):
        processes[i].join()

@exec_timer
def train_ray(cfg):
    workers = [RayWorker.remote(id=i) for i in range(cfg.n_interactors)]
    ray.get([worker.run.remote() for worker in workers])

class Config:
    def __init__(self) -> None:
        self.n_interactors = 5

if __name__ == "__main__":
    cfg = Config()
    ray.shutdown()
    ray.init()
    train_single(cfg)
    train_mp(cfg)
    train_ray(cfg)
```

执行会输出结果：
```bash
2023-10-02 15:46:01,351 INFO worker.py:1612 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
函数 train_single 执行时间：0.0006740093231201172 秒
函数 train_mp 执行时间：0.6658468246459961 秒
函数 train_ray 执行时间：0.6961688995361328 秒
```

可以看到，单进程执行时间最短，`multiprocessing`次之，`Ray`最长，这就是因为通信的基础消耗导致的。通过更改函数的执行时间和`n_interactors`参数，得到下表：



Ray不适合做同步，通信至少需要0.002s左右，只有每次任务执行时间够长或者通信量够大才能体现出Ray的优势，否则Ray的通信开销会比较大。

```python
import ray
import time
@ray.remote
class Worker:
    def __init__(self,id = 0) -> None:
        self.id = id
    def run(self):
        # 模拟任务执行
        for i in range(1000):
            self.id_back = self.id
        # time.sleep(1)
        return self.id
    
class BaseWorker:
    def __init__(self,id = 0) -> None:
        self.id = id
    def run(self):
        # 模拟任务执行
        for i in range(5000):
            self.id_back = self.id
        # time.sleep(1)
        return self.id
@ray.remote
class Learner:
    def __init__(self,id = 0) -> None:
        self.id = id 
    def run(self):
        # 模拟任务执行
        time.sleep(1)
        return self.id

def train(cfg,workers,leaner):
    # for i in range(cfg.n_interactors):
    #     workers[i].run.remote()
    s_t = time.time()
    print(ray.get([workers[i].run.remote() for i in range(cfg.n_interactors)]))
    e_t = time.time()
    print(f"worker: {e_t - s_t}")
    baseworker = BaseWorker()
    s_t = time.time()
    baseworker.run()
    e_t = time.time()
    print(f"base worker: {e_t - s_t}")
    print(ray.get(leaner.run.remote()))


class Config:
    def __init__(self) -> None:
        self.n_interactors = 5
if __name__ == "__main__":
    ray.shutdown()
    context = ray.init(
            #  local_mode = True,
             include_dashboard = True,
             dashboard_host="127.0.0.1",
             dashboard_port=8265)
    print(context.dashboard_url)
    cfg = Config()
    workers = [Worker.remote(id=i) for i in range(cfg.n_interactors)]
    leaner = Learner.remote()
    for i in range(10):
        s_t = time.time()
        train(cfg,workers,leaner)
        e_t = time.time()
        print(f"total time: {e_t - s_t}")
```

实验设备：2.6 GHz 六核Intel Core i7，MacOS 13.4.1 (c) (22F770820d)


|  BaseWorker.run   | worker数 | 单进程消耗(s) | mp消耗(s) | ray消耗(s) |
| :---------------: | :------: | :-----------: | :-------: | :--------: |
|    直接return     |    1     |    2.29e-5    |   0.572   |   0.540    |
|                   |    5     |    2.09e-5    |   0.548   |   0.683    |
|                   |    10    |    2.69e-5    |   0.994   |   1.099    |
| time.sleep(0.001) |    1     |    0.0001     |   0.439   |   0.498    |
|                   |    5     |    0.0007     |   0.544   |   0.617    |
|                   |    10    |    0.0014     |   0.847   |   0.980    |
|   time.sleep(1)   |    1     |    1.0014     |   1.444   |   1.515    |
|                   |    5     |     5.019     |   1.539   |   1.624    |
|                   |    10    |    10.037     |   1.975   |   1.915    |

从表中可以看出：

* 单进程消耗跟worker数成正比
* 当进程执行时间远大于通信消耗时，多进程的优势才会体现出来

给`RL`开发的启发：

* 当worker和learner顺序执行时，使用on-policy算法与环境交互会更有优势，因为on-policy算法每次会与环境交互若干个回合，这样每次执行的时间会比较长
* 单进程下使用vectorenv，尤其是对于one-step的算法（DQN等）会更有优势
* 多进程下所有的进程都保持异步，这样在简单环境中相较于单进程不至于太过弱势，在复杂环境中优势则更明显。
