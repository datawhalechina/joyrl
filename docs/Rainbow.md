## 算法参数说明

Rainbow的算法参数如下：

```python
class AlgoConfig(DefaultConfig):
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.tau = 1.0 # 1.0 means hard update
        self.hidden_dim = 256 # hidden_dim for MLP
        self.Vmin = 0. # support of C51  
        self.Vmax = 200. # support of C51 
        self.num_atoms = 51 # support of C51
        self.support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms) # support of C51
        self.delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1) # support of C51

        self.n_step = 1 #the n_step for N-step DQN
        self.batch_size = 32 # batch size
        self.lr = 0.0001 # learning rate
        self.target_update = 200 # target network update frequency
        self.memory_capacity = 10000 # size of replay buffer
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
```

参数配置和C51基本一致，其中增加的n_step表示n_step DQN的步长。同样，Vmin表示支撑中的最小值，Vmax表示支撑中的最大值，num_atoms表示支撑的单元数，support表示C51中的支撑表示分布可能取到的值，delta_z表示相邻支撑单元之间的差距。

剩下的参数都和DQN基本保持一致，这里不再赘述。