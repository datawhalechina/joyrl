## 算法参数说明

NoisyDQN的算法参数如下，基本和DQN中的一致：

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
```

其中gamma是强化学习中的折扣因子，一般调整在0.9-0.999之间即可，可以默认为0.99。buffer_size、target_update以及epsilon都需要根据实际环境的情况来经验性的调整。

NoisyDQN中的epsilon的衰减机制和DQN的保持一致。总体来说，NoisyDQN的参数和DQN大体一致，这里不再赘述。