The default parameter settings for the environment are stored in `joyrl/framework/envs/gym/config.py`, as follows:

### Q-learning

```python
class AlgoConfig:
    def __init__(self) -> None:
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 300 # epsilon decay rate
        self.gamma = 0.90 # discount factor
        self.lr = 0.1 # learning rate
```

Note:

* Set `epsilon_start=epsilon_end` can obtain fixed `epsilon=epsilon_end`.
* Adjust `epsilon_decay` appropriately to ensure that `epsilon` will not decay too early during the training process.
* Since the traditional reinforcement learning algorithm faces a relatively simple environment, `gamma` is generally set to `0.9`, and `lr` can be set to a relatively large value such as `0.1`, and there is no need to worry too much about overfitting.

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
        self.max_buffer_size = 100000  # size of replay buffer
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