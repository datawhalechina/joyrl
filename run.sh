# conda activate joyrl
python offline_run.py --yaml presets/ClassControl/Pendulum-v1/Pendulum-v1_DDPG.yaml
# python -m cProfile -o time.prof offline_run.py --yaml presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml 
# snakeviz output_file.prof

# mprof run offline_run.py --yaml presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml
# mprof plot

# python -m memory_profiler offline_run.py --yaml presets/ClassControl/CartPole-v1/CartPole-v1_DQN.yaml