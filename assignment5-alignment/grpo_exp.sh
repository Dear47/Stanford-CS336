#!/bin/bash

# 缓解碎片化显存 (对于 GRPO 这种涉及大量 Rollout 的算法很有必要)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定义日志目录
LOG_DIR="${LOG_DIR:-outputs/logs/grpo_execution_logs}"
DATANAME="MATH-500"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
mkdir -p $LOG_DIR/$DATANAME

echo "🚀 Starting GRPO Experiment Execution..."
echo "---------------------------------------------"

# ==========================================
# 实验 0: REINFORCE without Baseline (对照组)
# 说明: 使用传统的 REINFORCE 算法 (不带 Baseline)，作为 GRPO 的下界参考
# 配置: G=4, Loss=no_baseline, Steps=200
# ==========================================
# echo "[0/8] Running REINFORCE without Baseline (G=4, Steps=200)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 4 \
#     --loss_type no_baseline \
#     --steps 200 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id no_baseline_g4\
#     --use_std_normalization\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/reinforcement_without_baseline.log" 2>&1

# echo "✅ REINFORCE without Baseline Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 1: REINFORCE Baseline (对照组)
# 说明: 使用传统的 REINFORCE 算法 (带 Baseline)，作为 GRPO 的下界参考
# 配置: G=4, Loss=reinforce_with_baseline, Steps=200
# ==========================================
# echo "[1/8] Running REINFORCE Baseline (G=4, Steps=200)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 4 \
#     --loss_type reinforce_with_baseline \
#     --steps 200 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id baseline_g4\
#     --use_std_normalization\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/reinforcement_with_baseline.log" 2>&1

# echo "✅ REINFORCE Baseline Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 2: GRPO Standard (小 Group)
# 说明: 开启 GRPO Clip Loss，使用较小的 Group Size
# 配置: G=4, Loss=grpo_clip, Steps=200
# ==========================================
# echo "[2/8] Running GRPO Standard (G=4, Steps=200)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 4 \
#     --loss_type grpo_clip \
#     --steps 200 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id grpo_g4 \
#     --use_std_normalization\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/grpo_g4.log" 2>&1

# echo "✅ GRPO (G=4) Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 3: GRPO High Group (大 Group)
# 说明: 增加 Group Size，理论上能提供更稳定的 Advantage 估计
# 配置: G=8, Loss=grpo_clip, Steps=200
# ==========================================
# echo "[3/8] Running GRPO High Group (G=8, Steps=200)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 8 \
#     --loss_type grpo_clip \
#     --steps 200 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id grpo_g8\
#     --use_std_normalization\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/grpo_g8.log" 2>&1

# echo "✅ GRPO (G=8) Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 4: GRPO Long Training (长训练)
# 说明: 在大 Group 的基础上，增加训练步数以观察收敛性
# 配置: G=8, Loss=grpo_clip, Steps=400
# ==========================================
# echo "[4/8] Running GRPO Long Training (G=8, Steps=400)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 8 \
#     --loss_type grpo_clip \
#     --steps 400 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id grpo_long\
#     --use_std_normalization\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/grpo_long.log" 2>&1

# echo "✅ GRPO Long Training Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 5: GRPO without std normalization
# 说明: 实验 2 的基础上，取消 std normalization
# 配置: G=4, Loss=grpo_clip, Steps=200
# ==========================================
# echo "[5/8] Running GRPO without std normalization (G=4, Steps=200)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 4 \
#     --loss_type grpo_clip \
#     --steps 200 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id grpo_g4_no_std\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/grpo_g4_no_std.log" 2>&1

# echo "✅ GRPO (G=4) without std normalization Finished."
# echo "---------------------------------------------"

# ==========================================
# 实验 6: GRPO High Group without std normalization
# 说明: 实验 3 的基础上，取消 std normalization
# 配置: G=8, Loss=grpo_clip, Steps=200
# ==========================================
echo "[6/8] Running GRPO High Group without std normalization (G=8, Steps=200)..."
python ./cs336_alignment/grpo_experiment.py \
    --model_name Qwen2.5-Math-1.5B\
    --data_name $DATANAME\
    --groupsize 8 \
    --loss_type grpo_clip \
    --steps 200 \
    --train_device "$TRAIN_DEVICE"\
    --eval_device "$EVAL_DEVICE"\
    --exp_id grpo_g8_no_std\
    --lr 1e-6\
    > "$LOG_DIR/$DATANAME/grpo_g8_no_std.log" 2>&1

echo "✅ GRPO (G=8) without std normalization Finished."
echo "---------------------------------------------"

# ==========================================
# 实验 7: GRPO Long Training without std normalization
# 说明: 实验 4 的基础上，取消 std normalization
# 配置: G=8, Loss=grpo_clip, Steps=400
# ==========================================
# echo "[7/8] Running GRPO Long Training without std normalization (G=8, Steps=400)..."
# python ./cs336_alignment/grpo_experiment.py \
#     --model_name Qwen2.5-Math-1.5B\
#     --data_name $DATANAME\
#     --groupsize 8 \
#     --loss_type grpo_clip \
#     --steps 400 \
#     --train_device "$TRAIN_DEVICE"\
#     --eval_device "$EVAL_DEVICE"\
#     --exp_id grpo_long_no_std\
#     --lr 1e-5\
#     > "$LOG_DIR/$DATANAME/grpo_long_no_std.log" 2>&1

# echo "✅ GRPO Long Training without std normalization Finished."
# echo "---------------------------------------------"

# echo "🎉 All GRPO experiments completed! Check wandb for results."
