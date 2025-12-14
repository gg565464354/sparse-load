import os
import sys
# local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0,'/workspace/playground/libs')
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import importlib
import transformers.models.qwen2.modeling_qwen2
importlib.reload(transformers.models.qwen2.modeling_qwen2)
import argparse
import torch
from utils import *

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
  parser = argparse.ArgumentParser(description="Generate partial weight")
  parser.add_argument("--our_model_path", default=None, 
                      help='our OPT model')
  parser.add_argument("--skewing_matrix_path", default=None, 
                      help='path to skewing matrix')
  parser.add_argument("--model", default="facebook/opt-6.7b", 
                      help='model')
  parser.add_argument("--model_type", default = "opt", 
                      help='model arch (opt, llama,qwen2)')
  parser.add_argument("--partial_weight_ratio", required=False, default=0.1, 
                      help='Ours: partial weight ratio')
  parser.add_argument("--output", required=True, 
                      help='output directory to store result')
  return parser
    
def main():
    ## get arguments
    parser = process_options()
    args = parser.parse_args()
    print(f"Partial Weight Ratio: {args.partial_weight_ratio}")
    file_path = "./pg19_firstbook.txt"

    # fname = f"modeling_qwen2_ours_setup.py"
    fname = f"modeling_{args.model_type}_ours_setup.py"
    set_symlink(args.model_type, fname)

    if args.our_model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(args.our_model_path, trust_remote_code=True).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    
    print(f"Model config: {model.config}")
    if args.skewing_matrix_path is not None:
        A = torch.load(args.skewing_matrix_path).to('cuda').to(torch.float16)
        print("Skewing matrix shape:", A.shape)
        if args.model_type == 'llama':
            for layer_num, layer in enumerate(model.model.layers):
                layer.self_attn.skewing_matrix = A[layer_num]
        elif args.model_type == 'qwen2':
            for layer_num, layer in enumerate(model.model.layers):
                layer.self_attn.skewing_matrix = A[layer_num]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    prompt = ["The bartender refused to serve the patron because the patron was drunk.\n\nThe girl politely declined the hamburger because she was a vegetarian.\n\nThe spy discovered the enemy's location because the spy bugged the enemy's phone.\n\nI tossed the ball upwards therefore the ball hit the ceiling.\n\nThe rider fell to the ground because the bull bucked the rider.\n\nThe pair of students came under scrutiny by the teacher because the students both received excellent grades."]
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    if args.model_type == "opt":
        for layer in model.model.decoder.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)
    elif args.model_type == "llama":
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.partial_weight_ratio = float(args.partial_weight_ratio)
            if isinstance(attn.skewing_matrix, torch.Tensor) and attn.skewing_matrix.dim() >= 3:
                kv_heads_in_ckpt = getattr(attn, "num_kv_heads_in_ckpt", model.config.num_key_value_heads)
                if attn.skewing_matrix.size(0) == kv_heads_in_ckpt and attn.num_key_value_groups > 1:
                    attn.skewing_matrix = attn.skewing_matrix.repeat_interleave(attn.num_key_value_groups, dim=0)
    elif args.model_type == "qwen2":
        for layer in model.model.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)

    print("Start Generation")
    import inspect
    layer = 0
    # 根据模型类型选择正确的层路径
    if args.model_type == "opt":
        decoder_layers = model.model.decoder.layers
    else:
        decoder_layers = model.model.layers
    attn_class = decoder_layers[layer].self_attn.__class__
    print("Attention 类来自模块:", inspect.getfile(attn_class))
    model.generation_config = GenerationConfig.from_model_config(model.config)    
    generated_ids = model.generate(input_ids, max_new_tokens=1, min_new_tokens=1, use_cache=False)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    basepath = args.output + "/" + os.path.basename(os.path.normpath(args.model)) + "_%s"%(args.partial_weight_ratio)
    if not os.path.exists(basepath):
        os.system("mkdir -p %s"%(basepath))



    if args.model_type == "opt":
        for layer in range(len(model.model.decoder.layers)):
            partial_weight = model.model.decoder.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
    elif args.model_type == "llama":
        for layer in range(len(model.model.layers)):
            partial_weight = model.model.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
    elif args.model_type == "qwen2":
        for layer in range(len(model.model.layers)):
            partial_weight = model.model.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
    print("Done")
if __name__ == "__main__":
    main()
