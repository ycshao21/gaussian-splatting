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
port=2000

# Only one dataset specified here
declare -a run_args=(
  # >>> Mip-NeRF360 >>>
    "bicycle"
    "bonsai"
    "counter"
    "garden"
    # "kitchen"
    # "room"
    # "stump"
    # "truck"
  # <<< Mip-NeRF360 <<<
    
    # "train"
  )

# prune percentage for the first prune
declare -a prune_percents=(0.6) 
# decay rate for the following prune. The 2nd prune would prune out 0.5 x 0.6 = 0.3 of the remaining gaussian
declare -a prune_decays=(0.6)  
# The volumetric importance power. The higher it is the more weight the volume is in the Global significant
declare -a v_pow=(0.1)


# Prune types (TODO)
# "max_v_important_score" 
# "v_important_score"
# "k_mean" (TODO)
# "important_score"
declare -a prune_types=(
  "v_important_score"
  )

# Check that prune_percents and prune_decays arrays have the same length
if [ "${#prune_percents[@]}" -ne "${#prune_decays[@]}" ]; then
  echo "The number of prune percents does not match the number of prune decays."
  exit 1
fi

# 创建日志文件夹
log_dir="logs/lightgaussian"
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

# 创建输出文件夹
output_dir="output/lightgaussian"
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Loop over the arguments array
for arg in "${run_args[@]}"; do  # 遍历训练使用的数据集
  for i in "${!prune_percents[@]}"; do  # 遍历不同的 prune percentage
    prune_percent="${prune_percents[i]}"
    prune_decay="${prune_decays[i]}"
    vp="${v_pow[i]}"

    for prune_type in "${prune_types[@]}"; do  # 遍历不同的 prune type
      # Wait for an available GPU
      while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
          echo "GPU $gpu_id is available. Starting train_densify_prune.py with dataset '$arg', prune_percent '$prune_percent', prune_type '$prune_type', prune_decay '$prune_decay', and v_pow '$vp' on port $port"
          CUDA_VISIBLE_DEVICES=$gpu_id nohup python train_densify_prune.py \
            -s "$data_dir/$arg" \
            -m "$output_dir/$arg" \
            --prune_percent $prune_percent \
            --prune_decay $prune_decay \
            --prune_iterations 20000 \
            --v_pow $vp\
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
        
      done  # End loop over v_pow values
    done
  done
done

# Wait for all background processes to finish
wait
echo "All train_densify_prune.py runs completed."
