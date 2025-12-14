#!/usr/bin/env python3
"""
CachedHeavyRecentAttentionMasker padding策略使用示例

这个示例展示了如何使用不同的padding策略：
1. "fixed": 使用固定值-1作为padding（默认）
2. "least_important": 使用最不重要的token作为padding
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from playground.libs.transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载配置
    config = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
    config._attn_implementation = "eager"  # 确保使用我们的自定义attention实现
    
    print("=== CachedHeavyRecentAttentionMasker Padding策略对比 ===\n")
    
    # 示例1: 使用固定值-1作为padding (默认)
    print("1. 使用固定值-1作为padding策略:")
    model_fixed = Qwen2ForCausalLM(config, padding_strategy="fixed")
    model_fixed.to(device)
    print(f"   - 模型已创建，使用 padding_strategy='fixed'")
    print(f"   - 在这种模式下，缓存未命中时使用-1作为padding值")
    print(f"   - 优点: 简单、稳定、不会引入额外的噪声")
    print()
    
    # 示例2: 使用最不重要的token作为padding
    print("2. 使用最不重要的token作为padding策略:")
    model_least_important = Qwen2ForCausalLM(config, padding_strategy="least_important")
    model_least_important.to(device)
    print(f"   - 模型已创建，使用 padding_strategy='least_important'")
    print(f"   - 在这种模式下，缓存未命中时使用注意力权重最低的token作为padding")
    print(f"   - 优点: 可能提供更好的语义连续性")
    print()
    
    # 准备输入数据
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    text = "Hello, how are you doing today? I hope you are having a great day!"
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    print("3. 测试输入文本:")
    print(f"   '{text}'")
    print()
    
    # 运行推理测试
    print("4. 运行推理测试:")
    with torch.no_grad():
        # 固定padding策略
        print("   使用固定padding策略...")
        outputs_fixed = model_fixed(**inputs)
        print(f"   - 输出形状: {outputs_fixed.logits.shape}")
        
        # 最不重要token padding策略
        print("   使用最不重要token padding策略...")
        outputs_least_important = model_least_important(**inputs)
        print(f"   - 输出形状: {outputs_least_important.logits.shape}")
    
    print()
    print("5. 打印命中率统计:")
    print("   固定padding策略的命中率:")
    model_fixed.model.print_hit_rate_summary(detailed=False)
    
    print("   最不重要token padding策略的命中率:")
    model_least_important.model.print_hit_rate_summary(detailed=False)
    
    print("\n=== 策略选择建议 ===")
    print("- 使用 'fixed' 策略当:")
    print("  * 你想要最稳定和可预测的行为")
    print("  * 你不希望padding引入任何语义信息")
    print("  * 你在做性能基准测试")
    print()
    print("- 使用 'least_important' 策略当:")
    print("  * 你希望获得更好的语义连续性")
    print("  * 你的应用对生成质量更敏感")
    print("  * 你愿意承担轻微的计算复杂度增加")


if __name__ == "__main__":
    main() 