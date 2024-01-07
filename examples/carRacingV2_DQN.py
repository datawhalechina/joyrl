
import os
import sys
from os.path import dirname
try:
    dir_ = dirname(dirname(__file__))
except Exception as e:
    dir_ = dirname(dirname('__file__'))

if len(dir_) == 0:
    dir_ = dirname(os.getcwd())
    
print(f"dir_={dir_}")
sys.path.append(dir_)
import joyrl


if __name__ == "__main__":
    yaml_path = f"{dir_}/presets/Box2D/CarRacing-v2/DQN_carRacing-v2.yaml"
    # yaml_path = f"{dir_}/presets/Box2D/CarRacing-v2/DQN_carRacing-v2_test.yaml"
    joyrl.run(yaml_path = yaml_path)
