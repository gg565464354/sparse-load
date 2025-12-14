#!/usr/bin/env python3

import os
import sys

# 在任何其他导入之前执行这些操作
def force_custom_transformers():
    # 移除所有包含 transformers 的路径
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if 'transformers' not in p.lower()]
    
    # 添加自定义路径到最前面
    custom_path = "/workspace/playground/libs"
    if custom_path not in sys.path:
        sys.path.insert(0, custom_path)
    
    print(f"Python 路径已修改，自定义路径: {custom_path}")
    print(f"当前 sys.path 前3项: {sys.path[:3]}")
    
    # 清除已加载的模块
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('transformers')]
    for module in modules_to_remove:
        del sys.modules[module]
    
    # 清除导入缓存
    import importlib
    importlib.invalidate_caches()

# 在所有其他导入之前调用
force_custom_transformers()

# 现在导入其他模块
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
from utils import *

# 验证导入的模块
import transformers
print(f"实际使用的 transformers 路径: {transformers.__file__}")

### Parameters

def process_options():
  parser = argparse.ArgumentParser(description="Qwen-2 Model")
  parser.add_argument("--model", required=True, 
                      help='Qwen-2 model to load')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser

def setup_custom_model():
    """设置自定义模型并确保正确加载"""
    # 设置符号链接
    set_symlink("qwen2", "modeling_qwen2_orig.py")
    
    # 清除所有相关模块
    import sys
    import importlib
    
    modules_to_clear = [name for name in sys.modules.keys() 
                       if name.startswith('transformers')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    importlib.invalidate_caches()
    
    # 重新导入
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    return AutoModelForCausalLM, AutoTokenizer, AutoConfig

def main():
    parser = process_options()
    args = parser.parse_args()

    # 设置符号链接
    set_symlink("qwen2", "modeling_qwen2_orig.py")
    
    # 验证符号链接
    model_path = "/workspace/playground/libs/transformers/models/qwen2"
    if os.path.exists(f"{model_path}/modeling_qwen2.py"):
        print("符号链接创建成功")
        if os.path.islink(f"{model_path}/modeling_qwen2.py"):
            print(f"链接目标: {os.readlink(f'{model_path}/modeling_qwen2.py')}")
    else:
        print("符号链接创建失败")
        sys.exit(1)
    
    # 强制使用自定义 transformers 路径
    custom_transformers_path = "/workspace/playground/libs"
    
    # 移除所有现有的 transformers 相关路径
    sys.path = [p for p in sys.path if 'transformers' not in p.lower()]
    
    # 将自定义路径插入到最前面
    sys.path.insert(0, custom_transformers_path)
    
    # 清除所有已导入的 transformers 模块
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('transformers'):
            modules_to_clear.append(module_name)
    
    for module_name in modules_to_clear:
        del sys.modules[module_name]
    
    # 强制清除导入缓存
    import importlib
    importlib.invalidate_caches()
    
    # 现在重新导入
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    # 验证导入的模块路径
    import transformers
    print(f"Transformers 模块路径: {transformers.__file__}")
    
    # 继续模型加载...
    model_name = os.path.basename(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 使用自定义模块创建模型
    from transformers.models.qwen2 import modeling_qwen2 as custom_qwen2
    model = custom_qwen2.Qwen2ForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16,
        config=config
    ).cuda()
    
    # 验证模型类型
    first_layer = model.model.layers[0].self_attn
    print(f"实际使用的模型模块: {first_layer.__class__.__module__}")
    
    # 继续其余代码...
    
    # 验证是否使用了自定义模型
    first_layer = model.model.layers[0].self_attn
    print(f"实际使用的模型模块: {first_layer.__class__.__module__}")
    print(f"模型文件路径: {first_layer.__class__.__module__.__file__ if hasattr(first_layer.__class__.__module__, '__file__') else '未知'}")
    
    # 如果仍然不是自定义模块，停止执行
    # if 'modeling_qwen2_orig' not in first_layer.__class__.__module__:
    #     print("错误: 仍在使用标准 transformers 模块，未加载自定义模型")
    #     print("请检查模块导入顺序和路径设置")
    #     sys.exit(1)
    
    head_dim = model.model.layers[0].self_attn.head_dim
    
    # 使用config而不是模型属性来获取n_head
    n_head = config.num_attention_heads
    n_layer = config.num_hidden_layers

    ### Generation
    file_path = "./pg19_firstbook.txt"

    with open(file_path, 'r') as file:
        prompt = file.read()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]

    print("Start Generation")
    
    # 确保使用 torch.no_grad() 来避免梯度计算
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=1, min_new_tokens=1)
    
    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
    
    # 在生成后检查属性是否被创建
    print("检查生成后的属性...")
    first_layer = model.model.layers[0].self_attn
    print(f"生成后第一层是否有 rope_query: {hasattr(first_layer, 'rope_query')}")
    
    query_v = {}
    key_v = {}

    for i, layer in enumerate(model.model.layers):
        # 检查是否有rope_query属性
        if hasattr(layer.self_attn, 'rope_query'):
            query_v[str(i)] = layer.self_attn.rope_query
            key_v[str(i)] = layer.self_attn.rope_key
            
            # 添加调试信息：打印张量形状
            if i == 0:
                print(f"第一层 rope_query 形状: {layer.self_attn.rope_query.shape}")
                print(f"第一层 rope_key 形状: {layer.self_attn.rope_key.shape}")
                print(f"配置中的注意力头数: {n_head}")
        else:
            print(f"警告：层 {i} 没有 rope_query 属性，跳过...")
            # 添加调试信息：打印第一层的所有属性
            if i == 0:
                print(f"第一层 self_attn 的所有属性:")
                attrs = [attr for attr in dir(layer.self_attn) if not attr.startswith('_')]
                for attr in sorted(attrs):
                    print(f"  - {attr}")
                print(f"模型类型: {type(layer.self_attn)}")
                print(f"模型模块: {layer.self_attn.__class__.__module__}")
            continue

    ### Gen Skewing Matrix A
    A = torch.zeros(n_layer, n_head, head_dim, head_dim).to('cuda').to(torch.float16)
    
    if not query_v:
        print("错误：没有找到任何具有 rope_query 属性的层")
        sys.exit(1)
    
    for name in query_v:
        layer = int(name)
        query = query_v[name]
        key = key_v[name]
        
        print(f"处理层 {layer}, query 形状: {query.shape}, key 形状: {key.shape}")
        
        # 获取实际的头数（可能与配置不同）
        actual_n_heads = min(query.shape[1], key.shape[1], n_head)
        
        for head in range(actual_n_heads):
            in_q = query[0, head]
            in_k = key[0, head]
            uq, sq, vq = torch.svd(in_q.to(torch.float))
            uk, sk, vk = torch.svd(in_k.to(torch.float))
            s = sq * sk
            a = torch.zeros(head_dim, head_dim).to('cuda')
            _, ind = s.sort()
            r,c = a.shape
            A[layer, head] = a.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq).to(torch.float16)

    save_dir = args.output
    if not os.path.exists(save_dir):
        os.system(f"mkdir -p {save_dir}")
    torch.save(A, save_dir + "/" + model_name + ".pt")
    print('处理完成')

if __name__ == "__main__":
    main()
