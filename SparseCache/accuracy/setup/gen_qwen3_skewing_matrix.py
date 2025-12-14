import sys
import os
# local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, "/root/sparse-load/playground/libs")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import argparse
import torch
from utils import *

### Parameters

def set_symlink(model_type, fname):
    # model_path = "../transformers/src/transformers/models/" + model_type
    model_path = "/root/sparse-load/playground/libs/transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("../src/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    os.system(cmd)
    os.chdir(curr_dir)

def process_options():
  parser = argparse.ArgumentParser(description="qwen3 Model")
  parser.add_argument("--model", required=True, 
                      help='qwen3 model to load')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser

def main():
    parser = process_options()
    args = parser.parse_args()

    ### Model load
    set_symlink("qwen3", "modeling_qwen3_orig.py")

    model_name = os.path.basename(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    n_head = config.num_attention_heads
    head_dim = config.hidden_size // n_head
    
    # 获取 GQA 的分组数
    if hasattr(config, "num_key_value_heads"):
        num_groups = n_head // config.num_key_value_heads
    else:
        num_groups = 1 # 默认为 MHA
    n_layer = config.num_hidden_layers

    ### Generation
    # file_path = "./pg19_firstbook.txt"

    # with open(file_path, 'r') as file:
    #     prompt = file.read()

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[:, :2048]
    prompts = get_qasper_calibration_data(num_samples=20) 
    input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.cuda()[:, :2048]
    print("Start Generation")

    generated_ids = model.generate(input_ids, max_new_tokens = 1, min_new_tokens = 1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    query_v = {}
    key_v = {}

    for i, layer in enumerate(model.model.layers):
        query_v[str(i)] = layer.self_attn.rope_query
        key_v[str(i)] = layer.self_attn.rope_key

    ### Gen Skewing Matrix A
    A = torch.zeros(n_layer, n_head, head_dim, head_dim).to('cuda').to(torch.float16)
    for name in query_v:
        layer = int(name)
        query = query_v[name]
        key = key_v[name]
        

        for head in range(n_head):
            group_id = head // num_groups
            in_q = query[0, head]
            # in_k = key[0, head]
            in_k = key[0, group_id]
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

if __name__ == "__main__":
    main()
