#!/bin/bash
# 完整的测试流程

cd /root/sparse-load/SparseCache/accuracy/benchmark

echo "========================================"
echo "测试修复后的代码"
echo "========================================"
echo ""

# 清理旧输出
echo "1. 清理旧输出..."
rm -rf pred/llama-2-7b-inst-32k/fixed-vanilla-test/
echo "   ✓ 完成"
echo ""

# 测试vanilla模式（不启用infinigen）
echo "2. 运行vanilla模式测试..."
echo "   命令: python longbench_pred.py --model llama-2-7b-inst-32k --model_type llama --datasets qasper --name fixed-vanilla-test"
echo ""

python longbench_pred.py \
    --model llama-2-7b-inst-32k \
    --model_type llama \
    --datasets qasper \
    --name fixed-vanilla-test 2>&1 | head -50

echo ""
echo "========================================"
echo "3. 检查输出结果"
echo "========================================"
echo ""

if [ -f "pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl" ]; then
    echo "✓ 输出文件已生成"
    echo ""
    echo "第1条记录的pred字段："
    cat pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl | head -1 | python3 -c "import sys, json; print(json.load(sys.stdin)['pred'][:200])"
    echo ""
    echo ""
    echo "检查乱码特征..."
    if grep -q '<<<' pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl || \
       grep -q '111' pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl || \
       grep -q '666' pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl; then
        echo "❌ 仍然检测到乱码特征"
        echo ""
        echo "前3条记录："
        head -3 pred/llama-2-7b-inst-32k/fixed-vanilla-test/qasper.jsonl
    else
        echo "✅ 没有检测到明显的乱码特征！"
    fi
else
    echo "❌ 输出文件未生成"
fi

echo ""
echo "========================================"
echo "测试完成"
echo "========================================"
