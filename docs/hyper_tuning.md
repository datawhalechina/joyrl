
# 参数说明

`JoyRL`旨在让用户只需要通过调参就能进行相关的强化学习实践，主要的参数包括：

* 通用参数(`GeneralConfig`)：跟运行模式相关的参数，如算法名称`algo_name`、环境名称`env_name`、随机种子`seed`等等；
* 算法参数(`AlgoConfig`)：算法本身相关参数，也是用户需要调参的主要参数；
* 环境参数(`EnvConfig`)：环境相关参数，比如`gym`环境中的`render_mode`等；

`JoyRL`目前提供两种方式来设置参数。其一是通过设置对应的`config.py`文件(具体位置参考目录树)，这些文件中也有对应的参数说明，然后运行`python main.py`即可，笔者不推荐使用这种方式。其二是通过设置`yaml`文件，然后运行`python main.py --config [path to yaml file]`即可，笔者推荐使用这种方式，因为这种方式更加灵活，而且可以通过`yaml`文件来设置多个参数组合，从而进行多组实验。`JoyRL`中提供了一些预设的`yaml`文件，用户可以参考这些文件来设置自己的`yaml`文件，这些预设的`yaml`文件存放在`presets`文件夹下面。


## 通用参数说明

该部分讲述对应算法的超参数调整的一些经验。
## 传统强化学习算法

### Q-learning

```python
class AlgoConfig:
    def __init__(self) -> None:
        self.epsilon_start = 0.95 # epsilon 起始值
        self.epsilon_end = 0.01 # epsilon 终止值
        self.epsilon_decay = 300 # epsilon 衰减率
        self.gamma = 0.90 # 折扣因子
        self.lr = 0.1 # 学习率
```
参数说明：

* 设置`epsilon_start=epsilon_end`可以得到固定的`epsilon=epsilon_end`。

调参总结：

* 适当调整`epsilon_decay`以保证`epsilon`在训练过程中不会过早衰减。By [johnjim0816](https://github.com/johnjim0816)。
* 由于传统强化学习算法面对的环境都比较简单，因此`gamma`一般设置为`0.9`，`lr`且设置得比较大，不用太担心过拟合的情况。By [johnjim0816](https://github.com/johnjim0816)。

## DRL基础

### DQN

```python
class AlgoConfig(DefaultConfig):
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
```
其中value_layers设置部分请参考网络参数说明，这里略过。gamma是强化学习中的折扣因子，一般调整在0.9-0.999之间即可，可以默认为0.99。除了网络参数设置之外，DQN参数调整的空间较少。buffer_size、target_update以及epsilon都需要根据实际环境的情况来经验性的调整。

这里着重一下epsilon的衰减机制，也就是探索率相关，在JoyRL中目前是以指数方式衰减的，如下：
```python
self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
```
转成数学公式如下：

$$
\varepsilon = (\varepsilon_{start}-\varepsilon_{end}) * e ^{- \frac{sample\_count}{epsilon\_decay}} + \varepsilon_{end}
$$

训练开始的时候$sample\_count$等于0，则$\varepsilon = \varepsilon_{start}$，相当于$\varepsilon_{start}$的概率进行随机策略，随着$sample\_count$逐渐增大，指数项$e ^{- \frac{sample\_count}{epsilon\_decay}}$就会逐渐趋近于0，最后就会接近于$\varepsilon_{end}$，也就是较小的探索率。因此这里的$epsilon\_decay$是比较重要的，跟环境的每回合最大步数和读者设置的训练回合数有关，或者说跟训练预估的$总步数=环境的每回合最大步数*读者设置的训练回合数$有关，因此需要有一个合理的设置，不要让指数项太快地趋近于0，此时会导致没有进行足够的随机探索。也不要让指数项等到训练结束了或者说到达总步数了还没有趋近于0，此时会导致整个训练过程中随机探索的部分占比过大，影响算法的收敛。

调参总结：

* batch_size跟深度学习一样，一般都在64，128和256之间(太大了训练的物理机器吃不消)。By [johnjim0816](https://github.com/johnjim0816)。
* lr一般取0.0001～0.01之间，建议从小的lr开始尝试，过大的lr虽然能帮助策略在训练初期快速更新，但也很容易导致最终的收敛效果一般，loss长期保持震荡状态。 By [GeYuhong](https://github.com/GeYuhong)
* epsilon_decay和epsilon_start一般可以取500~800和0.95~0.99之间，通常希望算法在训练初期具有较强的探索能力，在后期逐渐稳定，对于某些特别需要探索的环境可以适当增大两者。By [GeYuhong](https://github.com/GeYuhong)