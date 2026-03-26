#!/bin/bash
# 通用工具函数：日志打印、参数校验、环境检查
# 被其他脚本引入使用

# ====================== 日志打印函数 ======================
# 彩色日志输出，方便区分不同级别
function log_info() {
    echo -e "\033[32m[INFO] $(date +'%Y-%m-%d %H:%M:%S'): $1\033[0m"
}

function log_warn() {
    echo -e "\033[33m[WARN] $(date +'%Y-%m-%d %H:%M:%S'): $1\033[0m"
}

function log_error() {
    echo -e "\033[31m[ERROR] $(date +'%Y-%m-%d %H:%M:%S'): $1\033[0m"
    exit 1
}

# ====================== 参数校验函数 ======================
# 检查文件是否存在
function check_file_exists() {
    local file_path=$1
    if [ ! -f "$file_path" ]; then
        log_error "文件不存在：$file_path"
    fi
}

# 检查目录是否存在，不存在则创建
function check_dir_exists() {
    local dir_path=$1
    if [ ! -d "$dir_path" ]; then
        log_info "创建目录：$dir_path"
        mkdir -p "$dir_path"
    fi
}

# ====================== 环境检查函数 ======================
# 检查Python环境
function check_python_env() {
    log_info "检查Python环境..."
    if ! command -v python3 &> /dev/null; then
        log_error "未找到python3，请先安装Python 3.8+"
    fi

    # 检查Python版本
    local python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    local major=$(echo $python_version | cut -d. -f1)
    local minor=$(echo $python_version | cut -d. -f2)

    if [ $major -lt 3 ] || [ $minor -lt 8 ]; then
        log_error "Python版本过低，需要3.8+，当前版本：$python_version"
    fi
    log_info "Python版本：$python_version ✅"
}

# 检查依赖是否安装
function check_dependencies() {
    log_info "检查依赖包..."
    if ! python3 -c "import torch, numpy, tokenizers, yaml" &> /dev/null; then
        log_warn "部分依赖未安装，正在安装..."
        pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    log_info "依赖检查通过 ✅"
}

# 检查GPU是否可用
function check_gpu() {
    log_info "检查GPU环境..."
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
        local gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        log_info "GPU可用：$gpu_count 张 | 型号：$gpu_name ✅"
        return 0
    else
        log_warn "未检测到GPU，将使用CPU训练（仅建议测试，训练速度极慢）"
        return 1
    fi
}

# ====================== 配置加载函数 ======================
# 从yaml配置文件读取参数（需要安装pyyaml）
function get_config_value() {
    local config_file=$1
    local key=$2
    # 使用Python读取yaml值，支持嵌套key（如：training.batch_size）
    python3 -c "
import yaml
with open('$config_file', 'r') as f:
    config = yaml.safe_load(f)
keys = '$key'.split('.')
value = config
for k in keys:
    value = value[k]
print(value)
"
}