## MOQ-learning

```python
class AlgoConfig:
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 300 # epsilon decay rate
        self.gamma = 0.90 # discount factor
        self.lr = 0.1 # learning rate
        self.weights = [0.5, 0.5] # weights for scalarization
```

其中gamma是强化学习中的折扣因子，一般调整在0.9-0.999之间即可，可以默认为0.99。weights为目标之间的权重向量；buffer_size、target_update以及epsilon都需要根据实际环境的情况来经验性的调整。

MOQ-Learning中的epsilon的衰减机制和DQN的保持一致。总体来说，MOQ-Learning的参数和DQN大体一致，这里不再赘述。
