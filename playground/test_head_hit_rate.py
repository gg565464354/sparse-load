#!/usr/bin/env python3
"""
测试修改后的头级命中率统计功能
"""

import sys
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig

# 添加libs路径
sys.path.insert(0, './libs')

from libs.transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

def test_head_hit_rate():
    """测试头级命中率统计功能"""
    print("=== 测试头级命中率统计功能 ===")
    
    # 配置和设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建一个简单的配置用于测试
    config = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B")
    config.num_hidden_layers = 2  # 只使用2层进行测试
    config.num_attention_heads = 4  # 4个注意力头
    
    print(f"测试配置: {config.num_hidden_layers}层, {config.num_attention_heads}个头")
    
    # 初始化模型
    try:
        model = Qwen2ForCausalLM(config)
        model.to(device)
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False
    
    # 创建测试输入
    batch_size = 1
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"测试输入形状: {input_ids.shape}")
    
    # 运行前向传播以生成一些统计数据
    try:
        model.eval()
        with torch.no_grad():
            # 多次前向传播以积累统计数据
            for i in range(5):
                outputs = model(input_ids)
                print(f"完成第 {i+1} 次前向传播")
        
        print("✓ 前向传播完成")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 检查统计数据
    try:
        # 获取统计数据
        stats = model.model.get_all_layers_hit_stats()
        print(f"✓ 获取到 {len(stats)} 层的统计数据")
        
        # 检查统计数据结构
        for i, stat in enumerate(stats):
            layer_idx = stat.get('layer_idx')
            head_stats = stat.get('head_stats', {})
            average_hit_rate = stat.get('average_hit_rate', 0.0)
            forward_count = stat.get('forward_count', 0)
            
            print(f"Layer {layer_idx}: {len(head_stats)} 个头, 平均命中率: {average_hit_rate:.2%}, Forward次数: {forward_count}")
            
            # 检查每个头的统计
            for head_idx, head_stat in head_stats.items():
                hit_tokens = head_stat.get('hit_tokens', 0)
                candidate_tokens = head_stat.get('candidate_tokens', 0)
                hit_rate = head_stat.get('hit_rate', 0.0)
                print(f"  Head {head_idx}: {hit_tokens}/{candidate_tokens} = {hit_rate:.2%}")
        
        print("✓ 统计数据结构正确")
        
    except Exception as e:
        print(f"✗ 统计数据检查失败: {e}")
        return False
    
    # 测试打印功能
    try:
        print("\n=== 测试详细统计输出 ===")
        model.model.print_hit_rate_summary(detailed=True)
        
        print("\n=== 测试汇总统计输出 ===")
        model.model.print_hit_rate_summary(detailed=False)
        
        print("✓ 打印功能正常")
    except Exception as e:
        print(f"✗ 打印功能失败: {e}")
        return False
    
    print("\n=== 所有测试通过! ===")
    return True

if __name__ == "__main__":
    success = test_head_hit_rate()
    if success:
        print("\n🎉 头级命中率统计功能工作正常!")
    else:
        print("\n❌ 测试失败，请检查代码修改。")
        sys.exit(1) 