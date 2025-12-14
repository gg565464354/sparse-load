import argparse
import json, tqdm
import torch
import copy
import os, sys
import math

def set_symlink(model_type, fname):
    # model_path = "../transformers/src/transformers/models/" + model_type
    model_path = "/workspace/playground/libs/transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("/workspace/SparseCache/accuracy/src/" + fname)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='opt')

    # Quant.
    parser.add_argument('--enable_quant', action='store_true')
    parser.add_argument("--qbits", type=int, default=8)

    # H2O
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    # InfiniGen
    parser.add_argument('--ours', action='store_true')
    parser.add_argument("--partial_weight_ratio", type=float, default=0.1)
    parser.add_argument("--partial_weight_path", type=str)
    parser.add_argument("--skewing_matrix_path", type=str)
    parser.add_argument("--alpha",type=float, default=5)
    parser.add_argument("--capacity",type=float, default=1.0)
    parser.add_argument("--budget",type=float, default=0.2)
    
    # Sparse-load
    parser.add_argument('--enable_heavy_hitter_masker', action='store_true')
    parser.add_argument("--heavy_budget_ratio", type=float, default=0.1)
    parser.add_argument("--recent_budget_ratio", type=float, default=0.0)
    parser.add_argument("--cache_mode", type=str, default="incremental", choices=["incremental", "async"])
    parser.add_argument("--cap_ratio", type=float, default=0.75)
    parser.add_argument("--apply_in_prefill", action="store_true")  # 不加就只建 cache 不稀疏 prefill
    parser.add_argument('--report-path', type=str, default=None, help='Path to save the cache hit report JSON file.')
    
    # [新增] Ark-vale 相关参数
    parser.add_argument('--enable_arkvale', action='store_true', help='Enable Ark-vale sparse algorithm')
    parser.add_argument('--arkvale_page_size', type=int, default=32, help='Page size for Ark-vale')
    parser.add_argument('--arkvale_page_budgets', type=int, default=128, help='Page budgets for Ark-vale (e.g., 4096 // 32). Set to -1 for full cache.')
    parser.add_argument('--arkvale_page_topks', type=int, default=32, help='Page top-k for Ark-vale')
    parser.add_argument('--arkvale_n_max_bytes', type=int, default=50 * (1 << 30), help='Max GPU memory in bytes for Ark-vale')
    parser.add_argument('--arkvale_n_max_cpu_bytes', type=int, default=60 * (1 << 30), help='Max CPU memory in bytes for Ark-vale')

    args = parser.parse_args()

    # [修改] 这里的逻辑保持不变，arkvale会使用原始的modeling文件，然后在模型加载后进行适配
    if args.enable_heavy_hitter_masker:
        set_symlink(args.model_type, f"modeling_{args.model_type}_cache.py")
    elif args.ours:
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
    else:
        # 默认或使用arkvale时，都使用原始的modeling文件
        set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")


    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    # import sys
    # sys.path.insert(0,'/workspace/playground/libs')
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
    if args.model_path is None:
        # if args.model_type == 'opt' or args.model_type == 'llama':
        if args.model_type == 'opt':
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16, attn_implementation="eager", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # [修改] 将原有的模型修改逻辑调整为 if/elif 结构，并加入 arkvale 的处理分支
    if args.enable_quant:
        if args.model_type == "opt":
            for i, layer in enumerate(model.model.decoder.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits
        if args.model_type == "llama":
            for i, layer in enumerate(model.model.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits
        if args.model_type == "qwen2":
            for i, layer in enumerate(model.model.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits
        if args.model_type == "qwen3":
            for i, layer in enumerate(model.model.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits

    elif args.enable_small_cache:
        from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
        from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
        ENABLE_Heavy_Hitter_FUNCTIONS = {
            "llama": convert_kvcache_llama_heavy_recent,
            "opt": convert_kvcache_opt_heavy_recent,
            "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
        }
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        base_path = os.path.basename(args.model_name)
        if not os.path.exists(f"../h2o_model/{base_path}.pt"):
            os.system("mkdir ../h2o_model")
            checkpoint = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, f"../h2o_model/{base_path}.pt")
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        model.load_state_dict(torch.load(f"../h2o_model/{base_path}.pt"))
        model = model.to(torch.float16)
    
    elif args.ours:
        if args.model_type == "opt":
            for layer in range(len(model.model.decoder.layers)):
                model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.decoder.layers[layer].self_attn.alpha = args.alpha
                model.model.decoder.layers[layer].self_attn.capacity = args.capacity
                model.model.decoder.layers[layer].self_attn.budget = args.budget
        if args.model_type == "llama":
            if args.skewing_matrix_path is not None:
                A = torch.load(args.skewing_matrix_path)
            for layer in range(len(model.model.layers)):
                model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.layers[layer].self_attn.alpha = args.alpha
                model.model.layers[layer].self_attn.capacity = args.capacity
                model.model.layers[layer].self_attn.budget = args.budget
                if args.skewing_matrix_path is not None:
                    model.model.layers[layer].self_attn.skewing_matrix = A[layer]
        if args.model_type == "qwen2":
            if args.skewing_matrix_path is not None:
                A = torch.load(args.skewing_matrix_path)
            for layer in range(len(model.model.layers)):
                model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.layers[layer].self_attn.alpha = args.alpha
                model.model.layers[layer].self_attn.capacity = args.capacity
                model.model.layers[layer].self_attn.budget = args.budget
                if args.skewing_matrix_path is not None:
                    model.model.layers[layer].self_attn.skewing_matrix = A[layer]
        if args.model_type == "qwen3":
            if args.skewing_matrix_path is not None:
                A = torch.load(args.skewing_matrix_path)
            for layer in range(len(model.model.layers)):
                model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.layers[layer].self_attn.alpha = args.alpha
                model.model.layers[layer].self_attn.capacity = args.capacity
                model.model.layers[layer].self_attn.budget = args.budget
                if args.skewing_matrix_path is not None:
                    model.model.layers[layer].self_attn.skewing_matrix = A[layer]

    # [新增] Ark-vale 逻辑
    elif args.enable_arkvale:
        print("Enabling Ark-vale sparse algorithm...")
        from arkvale import adapter
        
        # page_budgets=None 表示不进行驱逐和召回 (full cache)
        # 我们约定命令行传入 -1 来代表 None
        if hasattr(model, 'lm_head'):
                original_lm_head_forward = model.lm_head.forward
        else:
            original_lm_head_forward = None
        budgets = args.arkvale_page_budgets if args.arkvale_page_budgets != -1 else None

        adapter.enable_arkvale(
            model, 
            dtype=torch.float16, 
            device=model.device, # 直接从已加载的模型中获取设备
            page_size=args.arkvale_page_size,
            page_budgets=budgets,
            page_topks=args.arkvale_page_topks,
            n_max_bytes=args.arkvale_n_max_bytes,
            n_max_cpu_bytes=args.arkvale_n_max_cpu_bytes,
        )
        print("Ark-vale enabled successfully.")
        if original_lm_head_forward is not None:
                print("Restoring original lm_head.forward for perplexity evaluation.")
                model.lm_head.forward = original_lm_head_forward

    # 独立启用 CachedHeavyRecentAttentionMasker（与 --ours 无关）
    if args.enable_heavy_hitter_masker:
        if args.model_type == "opt":
            layers = model.model.decoder.layers
        else:
            layers = model.model.layers
        for layer in range(len(layers)):
            attn = layers[layer].self_attn
            attn.enable_heavy_mask = True
            attn.heavy_hitter_masker.heavy_budget_ratio = args.heavy_budget_ratio
            attn.heavy_hitter_masker.recent_budget_ratio = args.recent_budget_ratio
            attn.heavy_hitter_masker.cache_mode = args.cache_mode
            attn.heavy_hitter_masker.rebuild_cap_ratio = args.cap_ratio
            attn.heavy_hitter_masker.apply_in_prefill = args.apply_in_prefill

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []
    density=[]
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            logits = model(input_ids, use_cache=False).logits.log_softmax(dim=-1)
            if args.ours:
                density.append(model.get_density())

            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
            
            gold_indices = input_ids[:, 1:] # skip first
            logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
            
            result['result'] = {
                "choices": [
                    {
                        "text": prompt, 
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)
            
            if args.ours:
                if args.model_type == "opt":
                    for layer in model.model.decoder.layers:
                        layer.self_attn.previous_hidden_states = None
                if args.model_type == "llama":
                    for layer in model.model.layers:
                        layer.self_attn.previous_hidden_states = None
                if args.model_type == "qwen2":
                    for layer in model.model.layers:
                        layer.self_attn.previous_hidden_states = None
                if args.model_type == "qwen3":
                    for layer in model.model.layers:
                        layer.self_attn.previous_hidden_states = None

    # if args.ours:
    #     density = sum(density) / len(density) * 100
    #     retain_ratio = (1 - math.sqrt(1 - density / 100)) * 100
    #     #print("\ndensity: %.2f"%(density))
    #     print("retain ratio: %.2f\n"%(retain_ratio))

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    if args.model_type == "opt":
            report = model.get_cache_hit_report()
            report_data = {"cache_hit_report": report}
            
            # 1. 仍然打印到控制台，方便实时查看
            print("Cache Hit Report:")
            print(json.dumps(report_data, ensure_ascii=False, indent=2))
            
            # 2. 如果提供了报告路径，则将其保存到文件
            if args.report_path:
                with open(args.report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=2)
                print(f"\nCache hit report successfully saved to {args.report_path}")