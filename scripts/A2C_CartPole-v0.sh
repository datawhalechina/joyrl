# run A2C on CartPole-v0
codes_dir=$(dirname $(dirname $(readlink -f "$0"))) # "codes" path
python $codes_dir/A2C/main.py --device cpu