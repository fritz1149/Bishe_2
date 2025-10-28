# 使用示例

## ✅ 正确用法

### 1. 多卡分布式训练（推荐）

```bash
# 使用 4 张 GPU，不需要 --distributed 参数（默认启用）
torchrun --nproc_per_node=4 simsiam.py \
    --batch_size 256 \
    --train_dir /path/to/train \
    --test_dir /path/to/test \
    --epochs 100

# 使用 2 张 GPU
torchrun --nproc_per_node=2 simsiam.py \
    --batch_size 128 \
    --train_dir /path/to/train \
    --test_dir /path/to/test
```

### 2. 单卡训练

```bash
# 方式 1: 直接运行（会自动检测到没有 RANK 环境变量，使用单卡）
python simsiam.py \
    --batch_size 64 \
    --train_dir /path/to/train \
    --test_dir /path/to/test

# 方式 2: 明确禁用分布式
python simsiam.py \
    --no-distributed \
    --batch_size 64 \
    --train_dir /path/to/train \
    --test_dir /path/to/test

# 方式 3: 指定使用哪张 GPU
CUDA_VISIBLE_DEVICES=0 python simsiam.py \
    --batch_size 64 \
    --train_dir /path/to/train \
    --test_dir /path/to/test
```

### 3. 使用部分 GPU

```bash
# 只使用 GPU 0 和 2
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 simsiam.py \
    --batch_size 128 \
    --train_dir /path/to/train \
    --test_dir /path/to/test
```

### 4. 后台运行

```bash
# 使用 nohup
nohup torchrun --nproc_per_node=4 simsiam.py \
    --batch_size 256 \
    --train_dir /path/to/train \
    --test_dir /path/to/test \
    > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 5. 完整参数示例

```bash
torchrun --nproc_per_node=4 simsiam.py \
    --batch_size 256 \
    --epochs 100 \
    --workers 16 \
    --train_dir /datasets/train \
    --test_dir /datasets/test \
    --save_dir ./checkpoints \
    --vocab_path config/encryptd_vocab.txt \
    --hidden_size 768 \
    --num_layers 12 \
    --heads_num 12 \
    --feedforward_size 3072 \
    --max_seq_length 4096 \
    --proj_arch "2048-2048" \
    --pred_arch "512" \
    --base_lr 0.05 \
    --momentum 0.9 \
    --wd 1e-4 \
    --fix_pred_lr \
    --weak_sample_rate 0.5 \
    --knn_n 200 \
    --knn_t 0.2 \
    --log_freq 10 \
    --resume
```

## ❌ 错误用法（会导致错误）

### 错误 1: 在 torchrun 命令中使用 --distributed

```bash
# ❌ 错误！torchrun 不认识 --distributed
torchrun --nproc_per_node=4 simsiam.py --distributed --batch_size 256

# ✅ 正确！不需要 --distributed（默认启用）
torchrun --nproc_per_node=4 simsiam.py --batch_size 256
```

### 错误 2: torchrun 参数位置错误

```bash
# ❌ 错误！--nproc_per_node 必须在脚本名之前
torchrun simsiam.py --nproc_per_node=4 --batch_size 256

# ✅ 正确！torchrun 的参数在脚本名之前
torchrun --nproc_per_node=4 simsiam.py --batch_size 256
```

## 🔍 验证是否使用分布式

运行后应该看到类似输出：

### 多卡分布式（正确）
```
| distributed init (rank 0), gpu 0
| distributed init (rank 1), gpu 1
| distributed init (rank 2), gpu 2
| distributed init (rank 3), gpu 3
```

### 单卡模式
```
Not using distributed mode!
```

## 📊 监控命令

```bash
# 实时查看 GPU 使用
watch -n 1 nvidia-smi

# 查看进程
ps aux | grep simsiam

# 查看端口占用
lsof -i :29500

# 杀死训练进程
pkill -9 -f simsiam
```

## 🎯 快速测试

```bash
# 1. 测试单卡
python simsiam.py --batch_size 8 --epochs 1

# 2. 测试 2 卡
torchrun --nproc_per_node=2 simsiam.py --batch_size 16 --epochs 1

# 3. 测试 4 卡
torchrun --nproc_per_node=4 simsiam.py --batch_size 32 --epochs 1
```

## 💡 常用组合

### 开发/调试（快速迭代）
```bash
python simsiam.py \
    --batch_size 32 \
    --epochs 5 \
    --log_freq 1
```

### 正式训练（4 卡）
```bash
torchrun --nproc_per_node=4 simsiam.py \
    --batch_size 256 \
    --epochs 100 \
    --workers 16 \
    --resume
```

### 恢复训练
```bash
torchrun --nproc_per_node=4 simsiam.py \
    --batch_size 256 \
    --resume  # 自动从 checkpoint.pt 恢复
```

## 🔧 环境变量

有时需要设置这些环境变量：

```bash
# NCCL 调试
export NCCL_DEBUG=INFO

# 禁用 P2P（如果遇到通信错误）
export NCCL_P2P_DISABLE=1

# 指定网络接口
export NCCL_SOCKET_IFNAME=eth0

# 然后运行训练
torchrun --nproc_per_node=4 simsiam.py --batch_size 256
```

## 📝 注意事项

1. **`--distributed` 参数已移除**，改为 `--no-distributed`
   - 默认：启用分布式（如果有 RANK 环境变量）
   - 单卡：自动检测或使用 `--no-distributed`

2. **Batch Size 规则**：
   - torchrun 多卡：`--batch_size` 是总 batch（会自动分配到各卡）
   - 单卡：`--batch_size` 就是实际 batch size

3. **学习率自动缩放**：
   - 代码会根据 batch size 自动调整：`lr = base_lr * batch_size / 256`

4. **Workers 数量**：
   - 推荐：`GPU数量 × 4`
   - 例如 4 GPU：`--workers 16`









