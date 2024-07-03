# Linux 服务器无 Root 权限用户的环境配置方法
原仓库中含有 `environment.yml` 文件, 但这种方法经常出现环境配置失败的问题, 下面的方法是我尝试下来可行的一种.

由于没有 Root 权限, 所以先确保服务器中装有 g++, 如果没有的话需要在下面通过 conda 安装.

## 1. 克隆 GitHub 仓库
```bash
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive  # 注意：必须以 recursive 方式克隆
cd gaussian-splatting
```
如果克隆的时候没有添加 recursive 参数, 需要在工作区执行以下命令:
```sh
git submodule update --init --recursive
```

## 2. 创建 Conda 环境
```bash
conda create -n gaussian
conda activate gaussian  # 激活环境
conda install python=3.7.13   # 包含 pip 22.3.1
```

## 3. 安装所需依赖
```bash
conda install -c nvidia cuda-nvcc=11.6  # 安装 NVCC
conda install cudatoolkit=11.6  # 安装 CUDA Toolkit

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia  # 安装 PyTorch

pip install tqdm plyfile
```

## 4. 设置虚拟环境变量
在激活环境时自动添加 CUDA 相关环境变量:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
vim $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 写入以下内容
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

在离开环境时移除添加的环境变量：
```bash
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
vim $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# 写入以下内容
unset CUDA_HOME
unset PATH
unset LD_LIBRARY_PATH
```

重新激活环境，确定脚本是否生效：
```bash
conda deactivate
conda actiavte gaussian
$CUDA_HOME  # 检查是否添加成功
```

## 5. 安装 Submodules 中的依赖
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```