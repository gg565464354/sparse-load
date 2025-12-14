#!/bin/bash
# 完整的测试流程，确保使用最新的代码

cd /root/sparse-load/SparseCache/accuracy/benchmark

echo "================================"
echo "清理旧的输出文件"
echo "================================"
rm -rf pred/llama-2-7b-inst-32k/test-fixed/

echo ""
echo "================================"
echo "运行测试（vanilla模式，不启用infinigen）"
echo "================================"
python longbench_pred.py \
    --model llama-2-7b-inst-32k \
    --model_type llama \
    --datasets qasper \
    --name test-fixed \
    --e

echo ""
echo "================================"
echo "检查输出"
echo "================================"
if [ -f "pred/llama-2-7b-inst-32k/test-fixed/qasper.jsonl" ]; then
    echo "输出文件已生成"
    echo ""
    echo "前3行输出："
    head -3 pred/llama-2-7b-inst-32k/test-fixed/qasper.jsonl
    echo ""
    echo "检查是否有乱码特征..."
    if grep -q '<<<\\|111\\|666\\|```' pred/llama-2-7b-inst-32k/test-fixed/qasper.jsonl; then
        echo "❌ 仍然有乱码"
    else
        echo "✅ 没有检测到明显的乱码特征"
    fi
else
    echo "❌ 输出文件未生成"
fi

echo ""
echo "================================"
echo "测试完成"
echo "================================"
