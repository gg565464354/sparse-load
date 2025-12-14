#!/bin/bash
# ## 清理软连接
rm ./flexgen/flexgen/flex_opt.py 
rm ./flexgen/flexgen/pytorch_backend.py

# ## 运行我的代码
# ln -s /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/mycache/flex_opt.py  ./flexgen/flexgen/flex_opt.py 
# ln -s  /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/mycache/pytorch_backend.py  ./flexgen/flexgen/pytorch_backend.py


## 运行infinigen代码
ln -s /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/infinigen/flex_opt.py  ./flexgen/flexgen/flex_opt.py 
ln -s  /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/infinigen/pytorch_backend.py  ./flexgen/flexgen/pytorch_backend.py


# 设置要测试的 gpu-batch-size 值列表
batch_sizes=(1 2 4 8 16 32)
# batch_sizes=(32)
prompt_len=2000
output_len=20
# max_kv=500
max_kv=500

# 日志文件名
LOG_FILE="run_log.txt"

# 清空旧日志（如果存在）
> "$LOG_FILE"

# 循环执行
for batch_size in "${batch_sizes[@]}"; do
    echo "===== Running with --gpu-batch-size $batch_size =====" | tee -a "$LOG_FILE"
    
    TRANSFORMERS_OFFLINE=1 python -m flexgen.flex_opt \
        --model /NVME1/projects/qin/opt-6.7b \
        --path /NVME1/projects/qin/opt-6.7b \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size "$batch_size" \
        --num-gpu-batches 1 \
        --prompt-len "$prompt_len" \
        --gen-len "$output_len" \
        --warmup-input-path flexgen/pg19_firstbook.txt \
        --test-input-path flexgen/pg19_firstbook.txt \
        --alpha 4 \
        --partial-weight-ratio 0.2 \
        --max-num-kv "$max_kv" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo -e "\n\n" | tee -a "$LOG_FILE"
done

echo "所有执行完成，日志保存在 $LOG_FILE"

########################################### 开始运行 我的代码
rm ./flexgen/flexgen/flex_opt.py 
rm ./flexgen/flexgen/pytorch_backend.py

# ## 运行我的代码
ln -s /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/mycache/flex_opt.py  ./flexgen/flexgen/flex_opt.py 
ln -s  /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/mycache/pytorch_backend.py  ./flexgen/flexgen/pytorch_backend.py


## 运行infinigen代码
# ln -s /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/infinigen/flex_opt.py  ./flexgen/flexgen/flex_opt.py 
# ln -s  /NVME1/projects/qin/InfiniGen-main/speedup/flexgen/infinigen/pytorch_backend.py  ./flexgen/flexgen/pytorch_backend.py


# 设置要测试的 gpu-batch-size 值列表
# batch_sizes=(1)

# 日志文件名
LOG_FILE="run_log.txt"

# 清空旧日志（如果存在）
# > "$LOG_FILE"

# 循环执行
for batch_size in "${batch_sizes[@]}"; do
    echo "===== Running with --gpu-batch-size $batch_size =====" | tee -a "$LOG_FILE"
    
    TRANSFORMERS_OFFLINE=1 python -m flexgen.flex_opt \
        --model /NVME1/projects/qin/opt-6.7b \
        --path /NVME1/projects/qin/opt-6.7b \
        --percent 100 0 0 100 100 0 \
        --overlap false \
        --gpu-batch-size "$batch_size" \
        --num-gpu-batches 1 \
        --prompt-len "$prompt_len" \
        --gen-len "$output_len" \
        --warmup-input-path flexgen/pg19_firstbook.txt \
        --test-input-path flexgen/pg19_firstbook.txt \
        --alpha 4 \
        --partial-weight-ratio 0.2 \
        --max-num-kv "$max_kv" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo -e "\n\n" | tee -a "$LOG_FILE"
done

echo "所有执行完成，日志保存在 $LOG_FILE"