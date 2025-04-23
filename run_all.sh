GPU_NUM=$1
MODEL_PATH=$2
CUDA_VISIBLE_DEVICES=0,1,6,7 python test_latency.py -n $GPU_NUM -m $MODEL_PATH

# CUDA_VISIBLE_DEVICES=0,1,6,7 python test_latency.py -n $GPU_NUM -m /data/ubuntu/kris/checkpoints/Qwen__Qwen2.5-72B-Instruct

# CUDA_VISIBLE_DEVICES=1,2 python test_latency.py -n 2 -m /data/ubuntu/kris/checkpoints/Qwen__Qwen2.5-72B-Instruct-AWQ

# CUDA_VISIBLE_DEVICES=1,2 python test_latency.py -n 2 -m /data/ubuntu/kris/checkpoints/Qwen__Qwen2.5-72B-Instruct-AWQ