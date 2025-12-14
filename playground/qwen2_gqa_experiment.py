import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import torch.nn.functional as F

def analyze_qwen2_gqa_kv_cache_union_real(custom_heads_per_group=None):
    """
    分析Qwen2 GQA中每个head基于真实attention权重选择top 20%token时，
    group内KV cache并集的大小
    
    Args:
        custom_heads_per_group: 自定义每组head数量，如果为None则使用模型原始配置
    """
    
    print("=== Qwen2 GQA 真实KV缓存并集分析 ===\n")
    
    # 加载Qwen2模型和配置
    model_name = "/share/models/Qwen2-7B"
    print(f"正在加载模型: {model_name}")
    
    try:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True  # 确保在模型级别启用attention输出
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ 模型加载成功")
        
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_layers = config.num_hidden_layers
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    print(f"Query头数量: {num_q_heads}")
    print(f"Key-Value头数量: {num_kv_heads}")
    print(f"层数: {num_layers}")
    
    # 计算每个KV head对应的Query head数量
    original_heads_per_group = num_q_heads // num_kv_heads
    heads_per_kv_group = custom_heads_per_group if custom_heads_per_group is not None else original_heads_per_group
    
    print(f"原始每个KV组内的Query头数量: {original_heads_per_group}")
    if custom_heads_per_group is not None:
        print(f"自定义每个组内的Query头数量: {heads_per_kv_group}")
        # 计算新的组数
        num_groups = num_q_heads // heads_per_kv_group
        print(f"新的组数: {num_groups}")
    else:
        print(f"每个KV组内的Query头数量: {heads_per_kv_group}")
        num_groups = num_kv_heads
    
    # 测试不同序列长度
    sequence_lengths = [512, 1024, 2048, 4096]
    keep_ratio = 0.1  # 每个head保留20%的token
    
    # 准备一个长文本，用于生成不同长度的序列
    long_text = """
    The field of artificial intelligence has undergone remarkable transformations over the past decade, 
    with large language models emerging as one of the most significant breakthroughs in natural language processing. 
    These sophisticated neural networks, built upon the transformer architecture, have demonstrated unprecedented 
    capabilities in understanding and generating human-like text across a wide variety of tasks and domains.
    
    At the heart of these models lies the attention mechanism, a revolutionary approach that allows the network 
    to selectively focus on different parts of the input sequence when processing each element. This mechanism 
    has proven to be particularly effective in capturing long-range dependencies and contextual relationships 
    that were previously challenging for traditional sequential models to handle effectively.
    
    The transformer architecture, first introduced in the seminal paper "Attention Is All You Need," has become 
    the foundation for most state-of-the-art language models. The key innovation lies in the self-attention 
    mechanism, which computes attention weights between all pairs of positions in the input sequence, allowing 
    the model to directly model relationships between distant elements without the need for recurrent connections.
    
    However, as these models have grown in size and complexity, computational efficiency has become a critical 
    concern. The quadratic complexity of the attention mechanism with respect to sequence length poses significant 
    challenges for processing long sequences. This has led to the development of various optimization techniques, 
    including grouped query attention (GQA), which aims to reduce the computational overhead while maintaining 
    model performance.
    
    Grouped query attention represents an elegant solution to the memory and computational challenges associated 
    with large-scale attention mechanisms. By sharing key-value pairs across multiple query heads within each 
    attention group, GQA significantly reduces the memory footprint of the key-value cache while preserving 
    the model's ability to capture diverse attention patterns across different heads.
    
    The implications of these architectural innovations extend far beyond mere computational efficiency. They 
    enable the deployment of larger, more capable models in resource-constrained environments and pave the way 
    for even more sophisticated applications of artificial intelligence in various domains, from natural language 
    understanding to code generation and beyond.
    """ * 10  # 重复文本以确保足够长
    
    results = {}
    
    model.eval()
    with torch.no_grad():
        for seq_len in sequence_lengths:
            print(f"\n--- 序列长度: {seq_len} ---")
            
            # 生成指定长度的输入
            inputs = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=seq_len)
            input_ids = inputs["input_ids"].to(model.device)
            actual_seq_len = input_ids.shape[1]
            
            print(f"实际序列长度: {actual_seq_len}")
            
            if actual_seq_len < 10:  # 序列太短跳过
                continue
                
            tokens_per_head = max(1, int(actual_seq_len * keep_ratio))
            print(f"每个head保留的token数: {tokens_per_head}")
            
            # 运行模型获取attention权重
            try:
                outputs = model(input_ids, output_attentions=True, return_dict=True)
                print(f"  模型输出类型: {type(outputs)}")
                
                # 检查attention输出
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attentions = outputs.attentions
                    print(f"  获取到attention权重，层数: {len(attentions)}")
                    print(f"  第一层attention形状: {attentions[0].shape}")
                else:
                    print("  ❌ 没有获取到attention权重")
                    continue
                    
            except Exception as e:
                print(f"  模型推理失败: {e}")
                continue
            
            # 分析每一层的attention
            layer_results = []
            
            # 只分析前几层来节省时间和内存
            analyze_layers = min(28, len(attentions))
            print(f"  分析前 {analyze_layers} 层...")
            
            for layer_idx in range(analyze_layers):
                print(f"    处理第 {layer_idx + 1} 层...")
                
                # attentions[layer_idx] 形状: (batch_size, num_heads, seq_len, seq_len)
                attn_weights = attentions[layer_idx][0]  # 取第一个batch: (num_heads, seq_len, seq_len)
                print(f"    attention权重形状: {attn_weights.shape}")
                
                # 计算每个head的top-k token
                group_union_sizes = []
                
                for group_idx in range(num_groups):
                    # 获取这个组内的所有Query head
                    start_head = group_idx * heads_per_kv_group
                    end_head = min(start_head + heads_per_kv_group, num_q_heads)  # 防止越界
                    
                    union_indices = set()
                    
                    for head_idx in range(start_head, end_head):
                        try:
                            # 获取这个head的attention权重
                            head_attn = attn_weights[head_idx]  # (seq_len, seq_len)
                            
                            # 应用softmax和求和，就像你的代码逻辑
                            tmp_attn = F.softmax(head_attn, dim=-1, dtype=torch.float32)
                            tmp_sum = torch.sum(tmp_attn, dim=-2)  # 对每个token位置求和: (seq_len,)
                            
                            # 获取top-k token位置
                            _, current_topk = tmp_sum.topk(k=tokens_per_head, dim=-1)
                            
                            # 添加到并集
                            union_indices.update(current_topk.cpu().numpy().tolist())
                            
                        except Exception as e:
                            print(f"      处理head {head_idx}时出错: {e}")
                            continue
                    
                    group_union_sizes.append(len(union_indices))
                
                if group_union_sizes:  # 确保有有效结果
                    # 计算这一层的平均并集大小
                    avg_union_size = np.mean(group_union_sizes)
                    layer_results.append(avg_union_size)
                    print(f"    第 {layer_idx + 1} 层平均并集大小: {avg_union_size:.1f}")
            
            if not layer_results:
                print(f"  ❌ 序列长度 {seq_len} 没有有效结果")
                continue
                
            # 计算所有层的平均结果
            mean_union_size = np.mean(layer_results)
            std_union_size = np.std(layer_results)
            
            # 计算关键比例
            union_ratio = mean_union_size / actual_seq_len
            single_head_ratio = tokens_per_head / actual_seq_len
            expansion_factor = union_ratio / single_head_ratio if single_head_ratio > 0 else 0
            
            # 计算理论并集比例（基于组内head数量独立选择20%的理论期望）
            theoretical_union_ratio = 1 - (1 - keep_ratio) ** heads_per_kv_group
            theoretical_union_size = actual_seq_len * theoretical_union_ratio
            
            results[seq_len] = {
                'actual_seq_len': actual_seq_len,
                'mean_union_size': mean_union_size,
                'std_union_size': std_union_size,
                'union_ratio': union_ratio,
                'single_head_ratio': single_head_ratio,
                'theoretical_union_ratio': theoretical_union_ratio,
                'theoretical_union_size': theoretical_union_size,
                'expansion_factor': expansion_factor,
                'layer_results': layer_results
            }
            
            print(f"  ✓ 平均并集大小: {mean_union_size:.1f} ± {std_union_size:.1f} 个token")
            print(f"  ✓ 并集占总序列的比例: {union_ratio:.3f} ({union_ratio*100:.1f}%)")
            print(f"  ✓ 单个head的比例: {single_head_ratio:.3f} ({single_head_ratio*100:.1f}%)")
            print(f"  ✓ 理论并集比例: {theoretical_union_ratio:.3f} ({theoretical_union_ratio*100:.1f}%)")
            print(f"  ✓ 相对单head膨胀: {expansion_factor:.2f}x")
            print(f"  ✓ 相对理论值比例: {union_ratio/theoretical_union_ratio:.2f}")
    
    return results

def visualize_real_results(results):
    """可视化真实实验结果"""
    if not results:
        print("没有结果可视化")
        return
        
    seq_lens = list(results.keys())
    actual_seq_lens = [results[seq_len]['actual_seq_len'] for seq_len in seq_lens]
    union_sizes = [results[seq_len]['mean_union_size'] for seq_len in seq_lens]
    theoretical_union_sizes = [results[seq_len]['theoretical_union_size'] for seq_len in seq_lens]
    single_head_sizes = [results[seq_len]['actual_seq_len'] * 0.2 for seq_len in seq_lens]
    
    union_ratios = [results[seq_len]['union_ratio'] * 100 for seq_len in seq_lens]
    theoretical_ratios = [results[seq_len]['theoretical_union_ratio'] * 100 for seq_len in seq_lens]
    expansion_factors = [results[seq_len]['expansion_factor'] for seq_len in seq_lens]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 图1：并集大小对比
    ax1.plot(actual_seq_lens, union_sizes, 'bo-', label='Actual Union Size', linewidth=2, markersize=8)
    ax1.plot(actual_seq_lens, theoretical_union_sizes, 'go--', label='Theoretical Union Size', linewidth=2)
    ax1.plot(actual_seq_lens, single_head_sizes, 'r--', label='Single Head Size (20%)', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('KV Cache Size')
    ax1.set_title('KV Cache Union Size Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：比例对比
    ax2.plot(actual_seq_lens, union_ratios, 'bo-', label='Actual Union Ratio', linewidth=2, markersize=8)
    ax2.plot(actual_seq_lens, theoretical_ratios, 'go--', label='Theoretical Union Ratio', linewidth=2)
    ax2.axhline(y=20, color='r', linestyle='--', label='Single Head Ratio (20%)')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Ratio of Total Sequence (%)')
    ax2.set_title('Union Ratio Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3：膨胀倍数
    ax3.bar(range(len(seq_lens)), expansion_factors, color='orange', alpha=0.7)
    ax3.set_xlabel('Sequence Length')
    ax3.set_ylabel('Expansion Factor (vs Single Head)')
    ax3.set_title('Union Expansion Factor')
    ax3.set_xticks(range(len(seq_lens)))
    ax3.set_xticklabels(seq_lens)
    ax3.grid(True, alpha=0.3)
    
    # 图4：层间变化（选择中等长度）
    if seq_lens:
        mid_idx = len(seq_lens) // 2
        mid_seq_len = seq_lens[mid_idx]
        layer_results = results[mid_seq_len]['layer_results']
        ax4.plot(range(1, len(layer_results) + 1), layer_results, 'bo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Average Union Size')
        ax4.set_title(f'Union Size Across Layers (Length={mid_seq_len})')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qwen2_gqa_real_analysis.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为 'qwen2_gqa_real_analysis.png'")

def print_real_summary(results):
    """打印真实实验总结"""
    if not results:
        print("没有实验结果")
        return
        
    print("\n" + "="*60)
    print("GQA 真实KV缓存并集分析总结")
    print("="*60)
    
    print(f"实验配置:")
    print(f"  - 每个head保留token比例: 20% (基于真实attention权重)")
    print(f"  - Qwen2-7B配置: 28个Query头, 4个KV头")
    print(f"  - 每个KV组内Query头数量: 7")
    print(f"  - 测试序列长度: {list(results.keys())}")
    
    print(f"\n主要发现:")
    for seq_len in sorted(results.keys()):
        result = results[seq_len]
        print(f"  序列长度 {seq_len} (实际={result['actual_seq_len']}):")
        print(f"    实际并集大小: {result['mean_union_size']:.1f} ± {result['std_union_size']:.1f}")
        print(f"    单head大小(20%): {result['actual_seq_len'] * 0.2:.1f}")
        print(f"    理论并集大小: {result['theoretical_union_size']:.1f}")
        print(f"    实际并集占比: {result['union_ratio']:.1%}")
        print(f"    理论并集占比: {result['theoretical_union_ratio']:.1%}")
        print(f"    相对单head膨胀: {result['expansion_factor']:.2f}x")
        print(f"    相对理论值比例: {result['union_ratio']/result['theoretical_union_ratio']:.2f}")
    
    # 计算总体统计
    all_expansion_factors = [result['expansion_factor'] for result in results.values()]
    all_union_ratios = [result['union_ratio'] for result in results.values()]
    all_theoretical_ratios = [result['theoretical_union_ratio'] for result in results.values()]
    
    avg_expansion = np.mean(all_expansion_factors)
    avg_union_ratio = np.mean(all_union_ratios)
    avg_theoretical_ratio = np.mean(all_theoretical_ratios)
    
    print(f"\n整体统计:")
    print(f"  平均相对单head膨胀倍数: {avg_expansion:.2f}x")
    print(f"  平均实际并集占比: {avg_union_ratio:.1%}")
    print(f"  平均理论并集占比: {avg_theoretical_ratio:.1%}")
    print(f"  实际vs理论比值: {avg_union_ratio/avg_theoretical_ratio:.2f}")
    
    print(f"\n结论:")
    print(f"  基于真实attention权重，GQA中同组内7个Query头的top-20%并集")
    print(f"  平均占原序列的{avg_union_ratio:.1%}，理论期望为{avg_theoretical_ratio:.1%}")
    print(f"  实际结果约为理论值的{avg_union_ratio/avg_theoretical_ratio:.1%}")


def compare_different_group_sizes():
    """
    比较不同group大小（每组head数量）对KV cache并集的影响
    """
    print("\n" + "="*70)
    print("比较不同Group大小对KV缓存并集的影响")
    print("="*70)
    
    # 定义要测试的group大小
    # Qwen2-7B有28个Query头，所以可以测试的group大小为28的因子
    group_sizes_to_test = [1, 2, 4, 7, 14, 28]  # 28的因子
    
    print(f"测试的group大小: {group_sizes_to_test}")
    print(f"总Query头数: 28")
    
    # 只测试一个中等长度的序列以节省时间
    test_seq_length = 2048
    
    group_comparison_results = {}
    
    for group_size in group_sizes_to_test:
        print(f"\n--- 测试Group大小: {group_size} ---")
        
        # 运行分析
        results = analyze_qwen2_gqa_kv_cache_union_real(custom_heads_per_group=group_size)
        
        if results and test_seq_length in results:
            result = results[test_seq_length]
            group_comparison_results[group_size] = {
                'union_size': result['mean_union_size'],
                'union_ratio': result['union_ratio'],
                'theoretical_union_ratio': result['theoretical_union_ratio'],
                'expansion_factor': result['expansion_factor'],
                'num_groups': 28 // group_size
            }
            
            print(f"  ✓ 组数: {28 // group_size}")
            print(f"  ✓ 平均并集大小: {result['mean_union_size']:.1f}")
            print(f"  ✓ 并集占比: {result['union_ratio']:.3f} ({result['union_ratio']*100:.1f}%)")
            print(f"  ✓ 理论并集占比: {result['theoretical_union_ratio']:.3f} ({result['theoretical_union_ratio']*100:.1f}%)")
            print(f"  ✓ 相对单head膨胀: {result['expansion_factor']:.2f}x")
        else:
            print(f"  ❌ Group大小{group_size}测试失败")
    
    return group_comparison_results


def visualize_group_comparison(group_results, seq_length=2048):
    """可视化不同group大小的比较结果"""
    if not group_results:
        print("没有group比较结果可视化")
        return
        
    group_sizes = sorted(group_results.keys())
    union_ratios = [group_results[size]['union_ratio'] * 100 for size in group_sizes]
    theoretical_ratios = [group_results[size]['theoretical_union_ratio'] * 100 for size in group_sizes]
    expansion_factors = [group_results[size]['expansion_factor'] for size in group_sizes]
    num_groups = [group_results[size]['num_groups'] for size in group_sizes]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 图1：并集占比 vs Group大小
    ax1.plot(group_sizes, union_ratios, 'bo-', label='Actual Union Ratio', linewidth=2, markersize=8)
    ax1.plot(group_sizes, theoretical_ratios, 'ro--', label='Theoretical Union Ratio', linewidth=2)
    ax1.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Single Head Ratio (20%)')
    ax1.set_xlabel('Group Size (Heads per Group)')
    ax1.set_ylabel('Union Ratio (%)')
    ax1.set_title('Union Ratio vs Group Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(group_sizes)
    
    # 图2：膨胀倍数 vs Group大小
    ax2.plot(group_sizes, expansion_factors, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Group Size (Heads per Group)')
    ax2.set_ylabel('Expansion Factor (vs Single Head)')
    ax2.set_title('Expansion Factor vs Group Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(group_sizes)
    
    # 图3：组数 vs Group大小
    ax3.bar(range(len(group_sizes)), num_groups, color='orange', alpha=0.7)
    ax3.set_xlabel('Group Size (Heads per Group)')
    ax3.set_ylabel('Number of Groups')
    ax3.set_title('Number of Groups vs Group Size')
    ax3.set_xticks(range(len(group_sizes)))
    ax3.set_xticklabels(group_sizes)
    ax3.grid(True, alpha=0.3)
    
    # 图4：实际vs理论比值
    actual_vs_theoretical = [group_results[size]['union_ratio'] / group_results[size]['theoretical_union_ratio'] 
                           for size in group_sizes]
    ax4.plot(group_sizes, actual_vs_theoretical, 'mo-', linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Perfect Match (1.0)')
    ax4.set_xlabel('Group Size (Heads per Group)')
    ax4.set_ylabel('Actual / Theoretical Ratio')
    ax4.set_title('Actual vs Theoretical Union Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(group_sizes)
    
    plt.suptitle(f'Group Size Comparison Analysis (Seq Length={seq_length})', fontsize=16)
    plt.tight_layout()
    plt.savefig('qwen2_gqa_group_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nGroup比较图表已保存为 'qwen2_gqa_group_comparison.png'")


def print_group_comparison_summary(group_results):
    """打印group比较结果总结"""
    if not group_results:
        print("没有group比较结果")
        return
        
    print("\n" + "="*70)
    print("不同Group大小的KV缓存并集比较总结")
    print("="*70)
    
    print("配置信息:")
    print("  - 测试序列长度: 2048")
    print("  - 每个head保留token比例: 20%")
    print("  - 总Query头数: 28")
    print("  - 测试的Group大小: " + ", ".join(map(str, sorted(group_results.keys()))))
    
    print("\n详细结果:")
    print(f"{'Group大小':<8} {'组数':<6} {'并集占比':<10} {'理论占比':<10} {'膨胀倍数':<8} {'实际/理论':<10}")
    print("-" * 60)
    
    for group_size in sorted(group_results.keys()):
        result = group_results[group_size]
        ratio_comparison = result['union_ratio'] / result['theoretical_union_ratio']
        print(f"{group_size:<8} {result['num_groups']:<6} "
              f"{result['union_ratio']:.3f}     {result['theoretical_union_ratio']:.3f}     "
              f"{result['expansion_factor']:.2f}       {ratio_comparison:.3f}")
    
    print(f"\n关键发现:")
    
    # 找到效果最好和最差的group大小
    best_group = min(group_results.keys(), key=lambda x: group_results[x]['union_ratio'])
    worst_group = max(group_results.keys(), key=lambda x: group_results[x]['union_ratio'])
    
    print(f"  - 最小并集占比: Group大小{best_group} ({group_results[best_group]['union_ratio']:.1%})")
    print(f"  - 最大并集占比: Group大小{worst_group} ({group_results[worst_group]['union_ratio']:.1%})")
    
    # 分析趋势
    group_sizes = sorted(group_results.keys())
    ratios = [group_results[size]['union_ratio'] for size in group_sizes]
    
    print(f"  - Group大小从{group_sizes[0]}增加到{group_sizes[-1]}:")
    print(f"    并集占比从{ratios[0]:.1%}变化到{ratios[-1]:.1%}")
    
    if ratios[-1] > ratios[0]:
        trend = "增加"
    else:
        trend = "减少"
    print(f"    总体趋势: {trend}")
    
    print(f"  - 单head baseline: 20%")
    print(f"  - 理论最优(Group大小1): {group_results[1]['theoretical_union_ratio']:.1%}")
    print(f"  - 理论最差(Group大小28): {group_results[28]['theoretical_union_ratio']:.1%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2 GQA KV Cache Union Analysis")
    parser.add_argument("--compare-groups", action="store_true", 
                       help="Compare different group sizes instead of running original experiment")
    parser.add_argument("--group_size", type=int, default=None,
                       help="Custom group size for single experiment")
    args = parser.parse_args()
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        has_matplotlib = True
    except ImportError:
        print("未安装matplotlib，跳过图表生成")
        has_matplotlib = False
    
    if args.compare_groups:
        # 运行group比较实验
        print("开始运行不同Group大小比较实验...")
        group_results = compare_different_group_sizes()
        
        if group_results:
            if has_matplotlib:
                visualize_group_comparison(group_results)
            print_group_comparison_summary(group_results)
            print("\nGroup比较实验完成！")
        else:
            print("Group比较实验失败")
    else:
        # 运行原始实验或自定义group大小实验
        if args.group_size:
            print(f"开始运行自定义Group大小({args.group_size})的KV缓存并集分析实验...")
            results = analyze_qwen2_gqa_kv_cache_union_real(custom_heads_per_group=args.group_size)
        else:
            print("开始运行GQA 真实KV缓存并集分析实验...")
            results = analyze_qwen2_gqa_kv_cache_union_real()
        
        if results:
            # 生成可视化
            if has_matplotlib:
                visualize_real_results(results)
            
            # 打印总结
            print_real_summary(results)
            print(f"\n实验完成！")
        else:
            print("实验失败，请检查模型路径和环境配置")