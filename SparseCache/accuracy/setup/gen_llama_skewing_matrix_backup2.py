from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
import os
from utils import *

### Parameters

def process_options():
  parser = argparse.ArgumentParser(description="Llama-2 Model")
  parser.add_argument("--model", required=True, 
                      help='Llama-2 model to load')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser

def main():
    parser = process_options()
    args = parser.parse_args()

    ### Model load
    set_symlink("llama", "modeling_llama3_orig.py")

    model_name = os.path.basename(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    head_dim = model.model.layers[0].self_attn.head_dim
    n_head = model.model.layers[0].self_attn.num_heads
    n_layer = config.num_hidden_layers

# ... existing code ...

    ### Generation
    file_path = "./pg19_firstbook.txt"

    with open(file_path, 'r') as file:
        prompt = file.read()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]

    print("Start Generation")

    generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    query_v = {}
    key_v = {}

    for i, layer in enumerate(model.model.layers):
        query_v[str(i)] = layer.self_attn.rope_query
        key_v[str(i)] = layer.self_attn.rope_key

    # 添加调试信息
    print(f"配置中的头数: {n_head}")
    print(f"配置中的层数: {n_layer}")
    print(f"头维度: {head_dim}")
    
    # 检查第一层的张量形状
    first_layer_idx = "0"
    if first_layer_idx in query_v:
        print(f"第一层 rope_query 形状: {query_v[first_layer_idx].shape}")
        print(f"第一层 rope_key 形状: {key_v[first_layer_idx].shape}")
        
        # 获取实际的张量维度
        actual_n_head_query = query_v[first_layer_idx].shape[1] if len(query_v[first_layer_idx].shape) > 1 else 1
        actual_n_head_key = key_v[first_layer_idx].shape[1] if len(key_v[first_layer_idx].shape) > 1 else 1
        print(f"实际 Query 张量中的头数: {actual_n_head_query}")
        print(f"实际 Key 张量中的头数: {actual_n_head_key}")
        
        # 使用实际的张量维度而不是配置中的维度
        n_head = actual_n_head_key

    ### Gen Skewing Matrix A
    A = torch.zeros(n_layer, n_head, head_dim, head_dim).to('cuda').to(torch.float16)
    for name in query_v:
        layer = int(name)
        query = query_v[name]
        key = key_v[name]

        for head in range(n_head):
            in_q = query[0, head]
            in_k = key[0, head]
            uq, sq, vq = torch.svd(in_q.to(torch.float))
            uk, sk, vk = torch.svd(in_k.to(torch.float))
            s = sq * sk
            a = torch.zeros(head_dim, head_dim).to('cuda')
            _, ind = s.sort()
            r,c = a.shape
            A[layer, head] = a.scatter(-1, ind.unsqueeze(0).repeat(r,1), vq).to(torch.float16)

# ... existing code ...

    save_dir = args.output
    if not os.path.exists(save_dir):
        os.system(f"mkdir -p {save_dir}")
    torch.save(A, save_dir + "/" + model_name + ".pt")

if __name__ == "__main__":
    main()
