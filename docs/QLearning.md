
## 算法参数说明 

PPO算法参数如下：

```python
class AlgoConfig:
    def __init__(self) -> None:
       ## 设置 epsilon_start=epsilon_end 可以得到固定的 epsilon，即等于epsilon_end
        self.epsilon_start = 0.95 # epsilon 初始值
        self.epsilon_end = 0.01 # epsilon 终止值
        self.epsilon_decay = 300 # epsilon 衰减率
        self.gamma = 0.90 # 奖励折扣因子
        self.lr = 0.1 # 学习率
```

* 适当调整`epsilon_decay`以保证`epsilon`在训练过程中不会过早衰减。
* 由于传统强化学习算法面对的环境都比较简单，因此`gamma`一般设置为`0.9`，`lr`且设置得比较大，不用太担心过拟合的情况。
