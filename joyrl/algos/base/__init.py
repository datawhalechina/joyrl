
class BaseExp:
    def __init__(self,state=None, action=None, reward=None, next_state=None, terminated=None, **kwargs):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminated = terminated