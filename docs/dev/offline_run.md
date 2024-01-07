
### Offline Run

If you want to run from source code for debugging or other purposes, you can clone this repo:

```bash
git clone https://github.com/datawhalechina/joyrl.git
```

Then install the dependencies:

```bash
pip install -r requirements.txt
# if you have installed joyrl, you'd better uninstall it to avoid conflicts
pip uninstall joyrl
```

Then you can run the following command to train a DQN agent on CartPole-v1 environment.

```bash
python offline_run.py --yaml ./presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
```