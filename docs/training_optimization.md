# OpenPI pi0.5 PyTorch 训练性能优化文档

## 环境

| 项目 | 配置 |
|------|------|
| GPU | 8× NVIDIA Pro 6000 (Blackwell SM120, 95GB) |
| 互联 | PCIe Gen5 x16, 8卡 all-reduce ~40 GB/s |
| 框架 | PyTorch 2.11 + CUDA 12.8 |
| 模型 | pi0.5 (PaliGemma 2B + Gemma Expert 300M) |
| 训练 | DDP, BS256, single-node 8-GPU |

## 镜像与代码

- **容器镜像**: `openpi-perf-optimized:20260621 (base: nvcr.io/nvidia/pytorch:26.03-py3)`
  - 存储路径: `/pfs/pfs-iQ14no/images/openpi-perf-optimized_20260621.tar`
- **代码仓库**: https://github.com/songzhan01/openpi
  - Commit: `0c31a5d`
- **本地路径**: `/pfs/pfs-iQ14no/songzhan/openpi`

## 性能结果

| 配置 | step time | samples/s | vs baseline |
|------|-----------|-----------|-------------|
| 默认 baseline | 3.442s | 74.4 | — |
| **优化后 (BS256)** | **1.930s** | **132.6** | **+78%** |
| 优化后 (BS384, max throughput) | 2.855s | 134.5 | +81% |

精度验证: 50步内 loss 差异 < 0.0003, 平均差异 0.05%, 数值等价。

## 优化措施

### 1. 修复冗余 top-level GC（+24%）
原始代码同时开启了 top-level GC 和 per-layer GC，导致每层 forward 被重计算 2 次。
关闭 top-level GC 后仅 per-layer GC 生效，每层只重计算 1 次。

```bash
# 默认已关闭 (pytorch_gradient_checkpointing=False)
# 如需显式指定: --no-pytorch-gradient-checkpointing
```

### 2. torch.compile Gemma forward（+32%）
对 PaliGemmaWithExpertModel.forward 应用 torch.compile，覆盖 ~90% 的 fwd+bwd 计算。

```bash
--model.pytorch-compile-gemma-mode max-autotune-no-cudagraphs
```

关键修复:
- 移除 forward 中的 `print()` 语句 → 消除 graph break
- 移除 forward 中的属性赋值 → 消除 recompile

### 3. DDP find_unused_parameters=False（+0.8%）
训练时所有参数都被使用，无需遍历 autograd graph 查找未使用参数。

```bash
--no-ddp-find-unused-parameters
```

### 4. Fused AdamW（+3.2%）
将优化器从默认 AdamW 切换为 fused 模式，减少 kernel launch 开销。

```python
# 代码中已硬编码 fused=True
torch.optim.AdamW(..., fused=True)
```

## 启动命令

### 稳定推荐配置 (BS256, 132.6 sps)

```bash
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0

.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name <EXP_NAME> \
  --batch-size 256 \
  --no-ddp-find-unused-parameters \
  --ddp-bucket-cap-mb 25 \
  --model.pytorch-compile-gemma-mode max-autotune-no-cudagraphs \
  --pytorch-weight-path /pfs/pfs-iQ14no/xuwenjie04/VLA/pi_datasets/open_pi_05_base_model_pytorch \
  --assets-base-dir /pfs/pfs-iQ14no/songzhan/openpi/assets
```

### 环境变量

```bash
# 必需: SM120 Blackwell reduction kernel 修复
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0

# 可选: 持久化编译缓存 (避免 400s 冷启动)
export TORCHINDUCTOR_CACHE_DIR=/pfs/pfs-iQ14no/models/openpi/torchinductor_cache

# 数据相关
export HF_LEROBOT_HOME=/pfs/pfs-iQ14no/xuwenjie04/VLA/pi_datasets/hf_lerobot_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 复现步骤

### 1. 启动容器

```bash
# 从已保存镜像启动
docker load -i /pfs/pfs-iQ14no/images/openpi-perf-optimized_20260621.tar
docker run --gpus all --ipc=host --net=host \
  -v /pfs:/pfs \
  -w /pfs/pfs-iQ14no/songzhan/openpi \
  --name openpi-train \
  openpi-perf-optimized:20260621 (base: nvcr.io/nvidia/pytorch:26.03-py3) bash

# 或使用现有容器创建脚本
bash /pfs/pfs-iQ14no/songzhan/release.sh
```

### 2. 验证环境

```bash
cd /pfs/pfs-iQ14no/songzhan/openpi
git log --oneline -1  # 确认 commit 0c31a5d
nvidia-smi            # 确认 8 GPU 可见
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

### 3. 运行训练

```bash
source _env.sh
export TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
export TORCHINDUCTOR_CACHE_DIR=/pfs/pfs-iQ14no/models/openpi/torchinductor_cache

.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_repro_test \
  --batch-size 256 \
  --num-train-steps 30 \
  --log-interval 1 \
  --save-interval 999999 \
  --no-wandb-enabled \
  --no-ddp-find-unused-parameters \
  --ddp-bucket-cap-mb 25 \
  --model.pytorch-compile-gemma-mode max-autotune-no-cudagraphs \
  --pytorch-weight-path ${PYTORCH_WEIGHT_PATH} \
  --assets-base-dir ${ASSETS_BASE_DIR}
```

预期结果:
- Step 0: ~420s (编译 + autotune)
- Step 2+: ~1.93s/step
- 吞吐: ~132.6 samples/s
- 峰值显存: ~65.7 GB/GPU

## 已验证无效的优化方向

| 方向 | 原因 |
|------|------|
| SAC context_fn | torch.compile 下 policy 被忽略 |
| SDPA/FlashAttention | torch.compile 的 fusion 比 SDPA 更优 |
| FlexAttention | SM120 backward 不支持 head_dim=256 |
| gc_every_n=2 + max-autotune | 编译器 FakeTensor tracing bug |
| Activation offload (sync) | 不减少 GPU 峰值显存 |
| Activation offload (async) | Inductor stream codegen bug |
| 降 BS + 减 GC | 吞吐损失 > 减少重计算收益 |

## 后续优化方向

| 方向 | 预期收益 | 工程量 |
|------|----------|--------|
| Tensor Parallelism | +15% (消除 GC 重计算) | 2-4 周 |
| 多机扩展 | 线性扩展 | 中等 |
| 等待 PyTorch 修复 Inductor stream bug | +20% (offload 生效) | 0 (等上游) |

## 镜像保存

```bash
# 容器基础镜像: nvcr.io/nvidia/pytorch:26.03-py3
# Commit 后保存到 PFS:
docker commit profile-songzhan openpi-perf-optimized:20260621
docker save openpi-perf-optimized:20260621 | gzip > /pfs/pfs-iQ14no/images/openpi-perf-optimized_20260621.tar.gz
```
