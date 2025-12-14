from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
import os
from utils import *

def process_options():
    parser = argparse.ArgumentParser(description="Generate partial weight")
    parser.add_argument("--our_model_path", default=None, 
                      help='our OPT/Llama/Qwen model')
    parser.add_argument("--skewing_matrix_path", default=None, 
                      help='path to skewing matrix')
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", 
                      help='model name or path')
    # 1. 修改帮助文档，增加 qwen3
    parser.add_argument("--model_type", default="opt", 
                      help='model arch (opt, llama, qwen3)')
    parser.add_argument("--partial_weight_ratio", required=False, default=0.1, 
                      help='Ours: partial weight ratio')
    parser.add_argument("--output", required=True, 
                      help='output directory to store result')
    return parser
    
def main():
    ## get arguments
    parser = process_options()
    args = parser.parse_args()
    
    # 2. 动态生成文件名，确保你能加载 modeling_qwen3_ours_setup.py
    fname = f"modeling_{args.model_type}_ours_setup.py"
    
    # 注意：这里假设 utils.set_symlink 能处理 qwen3。
    # 如果你的 utils.py 里写死了只支持 llama/opt，你需要去 utils.py 里也改一下
    set_symlink(args.model_type, fname)

    # 加载模型
    if args.our_model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.our_model_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True # Qwen 系列通常建议加上这个
        ).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()

    # 3. 注入 Skewing Matrix
    # Qwen3 的结构是 model.model.layers，与 Llama 一致
    if args.skewing_matrix_path is not None:
        A = torch.load(args.skewing_matrix_path).to('cuda').to(torch.float16)
        
        if args.model_type in ['llama', 'qwen3']: # 将 qwen3 加入判断
            for layer_num, layer in enumerate(model.model.layers):
                # 确保 skewing matrix 维度匹配 (处理 GQA 情况)
                if hasattr(layer.self_attn, 'skewing_matrix'):
                    layer.self_attn.skewing_matrix = A[layer_num]
                else:
                    # 如果 model file 没替换成功，这里会报错，方便调试
                    print(f"Warning: Layer {layer_num} does not have skewing_matrix attribute.")
                    layer.self_attn.skewing_matrix = A[layer_num]

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        use_fast=False, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad_token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    # prompt = ["The bartender refused to serve the patron because the patron was drunk.\n\nThe girl politely declined the hamburger because she was a vegetarian.\n\nThe spy discovered the enemy's location because the spy bugged the enemy's phone.\n\nI tossed the ball upwards therefore the ball hit the ceiling.\n\nThe rider fell to the ground because the bull bucked the rider.\n\nThe pair of students came under scrutiny by the teacher because the students both received excellent grades."]
    # prompts = get_qasper_calibration_data(num_samples=20) 
    qasper_data = get_qasper_calibration_data(num_samples=20)
    input_ids = tokenizer(qasper_data, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.cuda()
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # 4. 设置 partial_weight_ratio
    if args.model_type == "opt":
        for layer in model.model.decoder.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)
    elif args.model_type in ["llama", "qwen3"]: # 将 qwen3 加入判断
        for layer in model.model.layers:
            layer.self_attn.partial_weight_ratio = float(args.partial_weight_ratio)

    print(f"Start Generation for {args.model_type}...")
    
    # 执行一次前向传播以触发 hook 计算 partial_weight
    generated_ids = model.generate(input_ids, max_new_tokens=1, min_new_tokens=1)

    print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    basepath = args.output + "/" + os.path.basename(os.path.normpath(args.model)) + "_%s"%(args.partial_weight_ratio)
    if not os.path.exists(basepath):
        os.system("mkdir -p %s"%(basepath))

    # 5. 保存 Partial Weight
    if args.model_type == "opt":
        for layer in range(len(model.model.decoder.layers)):
            partial_weight = model.model.decoder.layers[layer].self_attn.partial_weight_q
            torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
    elif args.model_type in ["llama", "qwen3"]: # 将 qwen3 加入判断
        for layer in range(len(model.model.layers)):
            # 这里的 .self_attn 必须对应 modeling_qwen3_ours_setup.py 里的定义
            if hasattr(model.model.layers[layer].self_attn, "partial_weight_q"):
                partial_weight = model.model.layers[layer].self_attn.partial_weight_q
                if partial_weight is not None:
                    torch.save(partial_weight, "%s/partial_weight_q_"%(basepath) + str(layer) + ".pt")
                else:
                    print(f"Layer {layer} partial_weight_q is None! Check if forward pass ran correctly.")
            else:
                print(f"Layer {layer} has no partial_weight_q attribute.")

if __name__ == "__main__":
    main()