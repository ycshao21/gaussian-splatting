#!/bin/bash

# 检查模型文件夹是否存在
data_dir="../data/MIP-NeRF360"
if [ ! -d "$data_dir" ]; then
  echo "The model directory '$data_dir' does not exist."
  exit 1
fi


# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


# Network GUI 端口号
port=1000

# Only one dataset specified here
declare -a run_args=(
  # >>> Mip-NeRF360 >>>
    # "bicycle"
    # "bonsai"
    # "counter"
    "flowers"
    # "garden"
    # "kitchen"
    # "room"
    # "stump"
    "treehill"
  # <<< Mip-NeRF360 <<<
    
    # "truck"
    # "train"
  )

# 创建日志文件夹
log_dir="logs/gaussian_splatting/original"
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

# 创建输出文件夹
output_dir="output/gaussian_splatting/original"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do  # 遍历训练使用的数据集

  # Wait for an available GPU
  while true; do
    gpu_id=$(get_available_gpu)
    if [[ -n $gpu_id ]]; then
      echo "GPU $gpu_id is available. Starting train_gaussian_splatting.py with dataset '$arg' on port $port"
      CUDA_VISIBLE_DEVICES=$gpu_id nohup python train_gaussian_splatting.py \
        -s "$data_dir/$arg" \
        -m "$output_dir/$arg" \
        --eval \
        --port $port > "$log_dir/train_$arg.log" 2>&1 &

      # Increment the port number for the next run
      ((port++))

      # Allow some time for the process to initialize and potentially use GPU memory
      sleep 60
      break
    else
      echo "No GPU available at the moment. Retrying in 1 minute."
      sleep 60
    fi
    done
        
  done
done

# Wait for all background processes to finish
wait
echo "All train_gaussian_splatting.py runs completed."
