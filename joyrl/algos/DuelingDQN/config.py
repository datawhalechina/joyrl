from config.config import DefaultConfig

class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.dueling = True # use dueling network
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate
        self.gamma = 0.99  # discount factor
        self.lr = 0.0001  # learning rate
        self.buffer_size = 100000  # size of replay buffer
        self.batch_size = 64  # batch size
        self.target_update = 4  # target network update frequency
        # value network layers config
        self.value_layers = [
            {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [256], 'activation': 'ReLU'},
        ]
