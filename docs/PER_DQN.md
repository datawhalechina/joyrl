## 算法参数说明

PER_DQN的算法参数如下，基本和DQN中的一致：

```python
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 500 # epsilon decay rate
        self.hidden_dim = 256 # hidden_dim for MLP
        self.gamma = 0.95 # discount factor
        self.lr = 0.0001 # learning rate
        self.buffer_size = 100000 # size of replay buffer
        self.batch_size = 64 # batch size
        self.target_update = 4 # target network update frequency
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}]
```


其中gamma是强化学习中的折扣因子，一般调整在0.9-0.999之间即可，可以默认为0.99。buffer_size、target_update以及epsilon都需要根据实际环境的情况来经验性的调整。

PER_DQN中的epsilon的衰减机制和DQN的保持一致。

因为PER_DQN只改变了replay buffer，这里的参数相比DQN基本变化不大。