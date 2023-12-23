# 快速开始

该部分主要讲述`JoyRL`的基本使用方法。

## 强化学习回顾

在强化学习中，智能体（agent）在环境（environment）中与环境进行交互，不断更新自身的策略，以获得最大化的奖励（reward），如下图：

<div align=center>
<img width="400" src="../figs/interaction_mdp.png"/>
</div>

在交互过程中会有四种元素：

* 智能体：负责与环境进行交互(`agent.sample(state)`)，并更新自身的策略(`agent.update(state, action, reward, next_state, done)`)。
* 环境：负责与智能体进行交互(`env.step(action)`)，并返回下一个状态(`next_state`)、奖励(`reward`)、是否结束(`done`)等信息。
* 经验池：负责存储智能体与环境交互的样本(`agent.buffer.push(state, action, reward, next_state, done)`)，并在训练时从中采样(`agent.buffer.sample(batch_size)`)，在`JoyRL`离线版中，我们将经验池封装到了智能体中，而在`JoyRL`在线版中，我们将经验池用额外的模块封装起来，具体参考`框架说明`部分。
* 交互过程：即智能体与环境交互的过程。

交互过程一般可以用伪代码描述，也可以称之为强化学习训练接口，不同的强化学习算法，其训练接口也不尽相同，但大体是相似的，如下：

```python
for i_episode in range(n_episodes):
    state = env.reset()
    for t in range(max_steps):
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        if done:
            break
```
[joyrl-book](https://github.com/datawhalechina/joyrl-book/tree/main/pseudocodes)提供了丰富的强化学习算法伪代码，帮助读者们更好地理解算法，也欢迎多多`star`～。在`JoyRL`中，我们将训练接口封装到了`Trainer`中，而在多线程中，我们将训练接口封装到了`Worker`中，具体参考相关说明部分。

## 定义环境

`JoyRL`中的环境主要有两种，一种是`gym`环境，一种是自定义环境，两者皆遵循`gym`接口。

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
``` 
## 定义策略

对于`DRL`算法，`JoyRL`中的策略主要包含两个元素，一个是网络，一个是数据处理器。

首先定义网络，例如在`DQN`算法中我们可以定义一个简单的网络：

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        pass
```
然后定义策略：

```python
class Policy:
    def __init__(cfg):
        self.policy_net = QNetwork(cfg.state_dim, cfg.action_dim) # 定义策略网络
        self.target_net = QNetwork(cfg.state_dim, cfg.action_dim) # 定义目标网络
    def get_action(self, state):
        ''' 根据状态采样动作
        '''
        pass
    def update(self):
        ''' 更新网络
        '''
```

然后需要定义数据处理器，即`DataHandler`，其作用主要是将与环境交互产生的样本转换成训练的样本，也是经验回放的载体，在`DQN`算法中，我们可以定义一个简单的数据处理器：

```python
class DataHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)() # create buffer
    def add_transition(self, transition):
        ''' 将与环境交互产生的样本添加到经验池中
        '''
        exp = self.create_exp(transition)
        self.buffer.push(exp)
    def sample_training_data(self):
        ''' 从经验池中采样训练数据
        '''
        exps = self.buffer.sample(self.cfg.batch_size)
        if exps is not None:
            return self.handle_exps_before_update(exps)
        else:
            return None
    def create_exp(self,transtion):
        ''' 将与环境交互产生的样本转换为经验
        '''
        state, action, reward, next_state, terminated, info = transtion
        exp_mod = __import__(f"algos.{self.cfg.algo_name}.exp", fromlist=['Exp'])
        exp = exp_mod.Exp(state = state, action = action, reward = reward, next_state = next_state, done = terminated, info = info)
        return [exp]
    def handle_exps_before_update(self, exps):
        ''' 将经验转换为训练数据
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
```

其中`JoyRL`中内置了一些常见的`Buffer`，用户可根据需要在`base/buffers.py`中自定义`Buffer`, 然后在`utils/core_types.py`中的`BufferType`类注册`Buffer`类型，即可在配置文件中使用。

## 定义训练器

基础的RL算法基本通过定义以上三种元素，即环境、策略和数据处理器，就可以进行训练了。对于更高级一些的算法，`JoyRL`提供更多的训练器，而对于多进程模式，训练器则以`Worker`和`Learner`的形式存在，具体参考相关说明部分。

## 超参数设置

在`JoyRL`中, 主要包含以下几类超参数：

* `general_cfg`: 通用超参数，包括算法名称`algo_name`，环境名称`env_name`，设备`device`，运行模式`mode`，随机种子`seed`等等
* `algo_cfg`: 算法超参数，包括网络结构配置、不同算法的参数等等
* `env_cfg`：环境超参数，例如`env_id`等等

用户可以通过`config\config.py`、`algos\[algo_name]\config.py`、`envs\[env_name]\config.py`来分别设置默认的通用超参数、算法超参数和环境超参数。并且在这些`py`文件中对各参数进行了注释说明。

此外，用户还可以通过`yaml`文件来设置超参数，如下：

```yaml
general_cfg:
  algo_name: DQN_new # algo name
  device: cpu # device, cpu or cuda
  env_name: gym # env name, differ from env_id in env_cfgs
  mode: train # run mode: train, test
  collect_traj: true
  mp_backend: single # multi-processing mode: null(default), ray
  n_interactors: 1 # number of workers if using multi-processing, default 1
  load_checkpoint: false # if load checkpoint or not
  load_path: Train_CartPole-v1_DQN_20221026-054757 # if load checkpoint, then config path in 'tasks' dir
  max_episode: 100 # max episodes, set -1 to keep running
  max_step: 200 # max steps per episode
  seed: 1 # random seed, set 0 not to use seed
  online_eval: true # if online eval or not
  online_eval_episode: 10 # online eval episodes
  model_save_fre: 500 # update step frequency of saving model
  save_best_model: # if save the best model or not, online_eval must be true)
  save_fig: true # if save fig or not
  show_fig: false # if show fig or not

algo_cfg:
  value_layers:
    - layer_type: linear
      layer_dim: [256]
      activation: relu
    - layer_type: linear
      layer_dim: [256]
      activation: relu
  batch_size: 64
  buffer_type: REPLAY_QUE
  buffer_size: 100000
  epsilon_decay: 500
  epsilon_end: 0.01
  epsilon_start: 0.95
  gamma: 0.95
  lr: 0.0001
  target_update: 4
env_cfg:
  id: CartPole-v1
  render_mode: null
```
## 运行

配置好默认参数之后，可以直接执行`python main.py`来运行。

也可以通过配置好`yaml`文件参数之后，执行`python main.py --yaml [yaml_file]`来运行，其中`[yaml_file]`是`yaml`文件的路径。

注意，`yaml`文件中的参数会覆盖`python`文件中的默认参数。

### 训练模式

设置参数：

```yaml
general_cfg:
  mode: train
```
即可开始训练。

如果想加载模型继续训练，可以设置参数：

```yaml
general_cfg:
  load_checkpoint: true # 加载模型
  load_path: Train_CartPole-v1_DQN_20221026-054757 # tasks文件夹下的模型路径
  mode: train
```

## 测试模式

设置参数：

```yaml
general_cfg:
  load_checkpoint: true # 加载模型
  load_path: Train_CartPole-v1_DQN_20221026-054757 # tasks文件夹下的模型路径
  mode: test
```

即可开始测试训练好的模型。

## 渲染模式

即可视化，通常用于测试模式。设置测试模式的参数之后，进一步设置参数

```yaml
env_cfg:
  render_mode: human # 渲染模式, None, human, rgb_array
```
其中`None`表示不渲染，`human`表示在屏幕上渲染，`rgb_array`表示返回渲染的图像。

当渲染模式为`rgb_array`时，会返回一个`numpy`数组，会在`[task_dir]/videos`下生成`video.gif`文件

## 在线测试模式

设置参数：

```yaml
general_cfg:
  online_eval: true # if online eval or not
  online_eval_episode: 10 # online eval episodes
  model_save_fre: 500 # update step frequency of saving model
```
在训练中，`JoyRL`会在每`model_save_fre`个更新步骤时保存模型，模型的名称为对应的`update_step`。

当开启在线测试模式时，即`online_eval`为`true`，此时会进行一次测试，测试的回合数为`online_eval_episode`，同时会额外保存一个`best`模型

## 收集模式

对于模仿学习、离线强化学习算法，需要收集轨迹，设置参数：

```yaml
general_cfg:
    collect_traj: true # if collect trajectories or not
```
运行结束时，会在`[task_dir]/results`下生成`trajs.pkl`文件，考虑到文件内存问题，每个`trajs.pkl`文件最多包含1000条轨迹, 超过1000条轨迹会生成多个`trajs.pkl`文件, 即`trajs_0.pkl`, `trajs_1.pkl`等等。

## 多进程模式

设置参数：

```yaml
general_cfg:
  mp_backend: ray # multi-processing mode: single(default), ray
  n_interactors: 4 # number of workers if using multi-processing, default 1
```
其中`single`表示普通的单进程模式，而多进程模式需要安装`ray`库，`n_interactors`表示进程数，进程数越多，训练速度越快。

关于多进程需要注意的地方：

* 单进程模式下，cuda不一定比cpu更快
* 对于简单的环境多进程不一定比单进程速度更快

第一点是因为当网络基本都是线性层时，cuda的并行计算能力并没有发挥出来。 第二点是因为简单的环境，例如`CartPole-v1`，每个`Worker`运行单回合时间很短，此时瓶颈主要在`Learner`这里，加上其他的通信开销等，反而会使得训练速度变慢。这种情况下，可以使用`multi-learner`模式(待开发)，以下是一个多进程瓶颈的简单实例：

```python
import ray
import time

@ray.remote
class Worker:
    def __init__(self,id) -> None:
        self.id = id
    def run(self,tracker,learners):
        while not ray.get(tracker.check_episode_limit.remote()):
            # print(f"curr_episode {ray.get(tracker.get_episode.remote())}")
            # 模拟单-learner
            ray.get(learners[0].learn.remote())
            # 模拟multi-learner
            # if self.id % 2 == 0:
            #     ray.get(learners[0].learn.remote())
            # else:
            #     ray.get(learners[1].learn.remote())
            ray.get(tracker.increase_episode.remote())
            time.sleep(0.1) # 模拟交互时间
@ray.remote
class Learner:
    def __init__(self) -> None:
        self.learn_count = 0
    def learn(self):
        self.learn_count += 1
        # time.sleep(0.1) # 模拟训练时间
@ray.remote            
class DataServer:
    def __init__(self):
        self.curr_episode = 0
        self.max_episode = 100
    def increase_episode(self):
        self.curr_episode += 1
    def get_episode(self):
        return self.curr_episode
    def check_episode_limit(self):
        return self.curr_episode > self.max_episode
        
if __name__ == "__main__":
    # 启动并行任务
    ray.shutdown()
    for n_interactors in [1,2,4]:
        ray.init()
        s_t = time.time()
        print(f"n_interactors {n_interactors}")
        workers = []
        for i in range(n_interactors):
            workers.append(Worker.remote(i))
        tracker = DataServer.remote()
        learners = []
        for i in range(2):
            learner = Learner.remote()
            learners.append(learner)
        worker_tasks = [worker.run.remote(tracker,learners) for worker in workers]
        # 等待任务完成
        ray.get(worker_tasks)
        e_t = time.time()
        print("time cost: ",e_t-s_t)
        ray.shutdown()
```

首先，注释掉`Learner`中的`time.sleep(0.1)`，运行结果如下：

```bash
n_interactors 1
time cost:  10.974063396453857
n_interactors 2
time cost:  5.955186367034912
n_interactors 4
time cost:  3.4605765342712402
```
可以看到随着`n_interactors`数增加，计算时间会减少，这是因为`Learner`的计算时间占比较小，而`Worker`的交互时间占比较大，多进程可以减少交互时间。

接着，注释掉`Worker`中的`time.sleep(0.1)`，并取消注释`Learner`中的`time.sleep(0.1)`，运行结果如下：

```bash
n_interactors 1
time cost:  11.051950216293335
n_interactors 2
time cost:  11.00266146659851
n_interactors 4
time cost:  11.287662506103516
```
可以看到随着`n_interactors`数增加，计算时间并没有减少，约维持在`10s`左右，这是因为`Learner`的计算时间占比较大，而`Worker`的交互时间占比较小，因此该程序的瓶颈主要在于`Learner`，多进程并不能加速训练。

但是当我们进一步开启`multi-learner`模式，见`Worker`中相关代码，运行结果如下：

```bash
n_interactors 1
time cost:  11.067777156829834
n_interactors 2
time cost:  5.9282824993133545
n_interactors 4
time cost:  6.0641443729400635
```
我们发现会发现`n_interactors`数目增加到2时，计算时间开始减少，但进一步增加`n_interactors`时，时间瓶颈维持在`5s`左右，这是因为只模拟了两个`Learner`。

综合以上实验，我们可以得出结论，需要根据交互时间和训练时间综合考虑多进程的设置方式，当然对于复杂的环境尤其是需要图像的Atari环境或者一些`on-policy`算法(这类算法并不会每步都更新算法，而是搜集一定的样本在更新算法，相当于`Worker`的交互时间占比更多)，多进程一般都能加速训练。