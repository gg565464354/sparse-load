#!/bin/bash
# 验证代码修复是否正确应用

echo "========================================"
echo "验证代码修复"
echo "========================================"
echo ""

FILE="/root/sparse-load/SparseCache/accuracy/benchmark/source/modeling_llama_ours.py"

echo "1. 检查 apply_rotary_pos_emb 函数..."
if grep -q "Check if position_ids are within bounds" "$FILE"; then
    echo "   ✅ 已添加边界检查"
else
    echo "   ❌ 未找到边界检查"
fi

if grep -q "position_ids.clamp" "$FILE"; then
    echo "   ❌ 仍然有clamp操作（应该删除）"
else
    echo "   ✅ clamp操作已删除"
fi
echo ""

echo "2. 检查 cos/sin 扩展逻辑..."
if grep -q "max_pos_id = position_ids.max" "$FILE"; then
    echo "   ✅ 已添加动态扩展"
else
    echo "   ❌ 未找到动态扩展逻辑"
fi
echo ""

echo "3. 检查 attention_mask 切片..."
if grep -q "causal_mask = attention_mask\[:, :, :, :kv_seq_len\]" "$FILE"; then
    echo "   ✅ 已添加attention_mask切片"
else
    echo "   ❌ 未找到attention_mask切片"
fi
echo ""

echo "4. 检查符号链接..."
LINK="/root/sparse-load/playground/libs/transformers/src/transformers/models/llama/modeling_llama.py"
if [ -L "$LINK" ]; then
    TARGET=$(readlink -f "$LINK")
    echo "   符号链接指向: $TARGET"
    if [ "$TARGET" = "$FILE" ]; then
        echo "   ✅ 符号链接正确"
    else
        echo "   ⚠️  符号链接指向其他文件"
    fi
else
    echo "   ❌ 不是符号链接"
fi
echo ""

echo "5. 检查 longbench_pred.py 模块重载逻辑..."
if grep -q "Force reload the transformers.models" "/root/sparse-load/SparseCache/accuracy/benchmark/longbench_pred.py"; then
    echo "   ✅ 已添加模块重载逻辑"
else
    echo "   ❌ 未找到模块重载逻辑"
fi
echo ""

echo "========================================"
echo "验证完成"
echo "========================================"
