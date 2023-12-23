
## 算法参数说明 

PPO算法参数如下：

```python
class AlgoConfig:
    def __init__(self):
        ppo_type = 'clip' # clip or kl
        self.gamma = 0.99 # discount factor
        self.k_epochs = 4 # update policy for K epochs
        self.actor_lr = 0.0003 # learning rate for actor
        self.critic_lr = 0.001 # learning rate for critic
        self.eps_clip = 0.2 # clip parameter for PPO
        self.entropy_coef = 0.01 # entropy coefficient
        self.update_freq = 100 # update policy every n steps
        self.actor_hidden_dim = 256 # hidden dimension for actor
        self.critic_hidden_dim = 256 # hidden dimension for critic
        # batch size
        self.train_batch_size = 100 # ppo train batch size
        self.sgd_batch_size = 64 # sgd batch size
        # continuous PPO
        self.continuous = False # continuous action space
        # KL parameter
        self.kl_target = 0.1 # target KL divergence
        self.kl_lambda = 0.5 # lambda for KL penalty, 0.5 is the default value in the paper
        self.kl_beta = 1.5 # beta for KL penalty, 1.5 is the default value in the paper
        self.kl_alpha = 2 # alpha for KL penalty, 2 is the default value in the paper
```

* `ppo_type`: PPO有两种Loss函数更新方式：clip方法和KL散度。现在一般都用clip方法更新，一方面因为KL调参比较费劲，另一方面clip方法基本可以满足所有需求
* `eps_clip`：clip参数，一般设置为0.1-0.2之间即可
* `entropy_coef`：策略熵损失系数，增加该系数提高actor的稳定性，保持0.0001-0.02即可，或者直接设置为0在一些问题中影响也不大
* `update_freq`：更新频率，在JoyRL中设置为每隔几步更新，一般跟环境中每回合最大步数线性相关，例如carpole-v1环境中每回合最大步数是200，这里更新频率可以设置为50，100，200等等，这项参数需要根据实际经验调整
* `k_epochs`：调整每次更新的epoch数，不能太大也不能太小，太大了一方面收敛速度会变慢，另一方面容易过拟合，太小了容易欠拟合
* `train_batch_size`: 一般取值比较大（这里取100实际上是为了计算简便），当batch_size比较大时，训练的结果比较准确，但是训练速度比较慢
* `sgd_batch_size`: 小批量样本，一般取值64或128。当batch_size特别小的时候，训练速度很快，但是训练结果准确性不高，这时就需要一个折中的办法，即使用小批量样本计算
* `continuous`: 动作空间是否连续
* `kl_target`: KL散度的目标值
* `kl_lambda`: KL惩罚项的系数，PPO论文中的默认值是0.5
* `kl_beta`: KL散度目标值的系数，默认值为1.5
* `kl_alpha`: KL惩罚项的系数的更新参数，默认值为2
