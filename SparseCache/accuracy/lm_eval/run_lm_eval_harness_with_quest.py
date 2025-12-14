import argparse
import json, tqdm
import torch
import copy
import os, sys
import math

def set_symlink(model_type, fname):
    # model_path = "../transformers/src/transformers/models/" + model_type
    model_path = "/workspace/playground/libs/transformers/src/transformers/models/" + model_type
    # model_path = "/opt/conda/lib/python3.11/site-packages/transformers/models/" + model_type
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
    
    #Sparse-load
    parser.add_argument('--enable_heavy_hitter_masker', action='store_true')
    parser.add_argument("--heavy_budget_ratio", type=float, default=0.1)
    parser.add_argument("--recent_budget_ratio", type=float, default=0.0)
    #Quest
    parser.add_argument('--quest', action='store_true')
    args = parser.parse_args()
    if args.enable_heavy_hitter_masker:
        set_symlink(args.model_type, f"modeling_{args.model_type}_test.py")
    elif args.ours:
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
        # set_symlink(args.model_type, f"modeling_{args.model_type}_ours_for_llama3.py")
    elif args.quest:
        set_symlink(args.model_type, f"modeling_{args.model_type}_quest_query_merge_recent.py")
    else:
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
        if args.model_type == 'opt' or args.model_type == 'llama':
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map = None, torch_dtype=torch.float16, attn_implementation="eager", trust_remote_code=True).eval().to("cuda")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map = 'auto', torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if args.quest:
        config.use_quest_sparse = True
        config.inference_batch_size = 1               # 你的 QuestManager 以 batch 为单位初始化
        config.quest_page_size = 16                   # 按你实现的缺省
        config.quest_top_k = 5                        # 可调
        config.quest_merge_mode = "kv"                # "kv" | "single" | "none"
        config.quest_merge_reduce = "mean"            # "mean" | "sum" | "maxabs" | "l2"

        # 让评测脚本的 --recent_ratio 直接作用到 Quest
        config.quest_recent_ratio = 0.1
        config.quest_recent_tokens = 0                # 比例优先，固定数置 0
        config.quest_recent_cap_pages = 8             # 可控上限，防止显存暴涨
    # if args.quest:
    #     model.quest_init(page_size=16, max_seq_len=32768, token_budget=2048, dtype=torch.float16, device=torch.device("cuda:0"))
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
    
    elif args.ours or args.enable_heavy_hitter_masker:
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

    # 独立启用 CachedHeavyRecentAttentionMasker（与 --ours 无关）
     # 独立启用 CachedHeavyRecentAttentionMasker（与 --ours 无关）
    if args.enable_heavy_hitter_masker:
        if args.model_type == "opt":
            layers = model.model.decoder.layers
        else:
            layers = model.model.layers
        for layer in range(len(layers)):
            attn = layers[layer].self_attn
            attn.enable_heavy_mask = True
            attn.sparse_selector.heavy_budget_ratio = args.heavy_budget_ratio
            attn.sparse_selector.recent_budget_ratio = args.recent_budget_ratio

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    results = []
    density=[]
    all_requests_stats = []
    total_tokens = 0
    for r in requests:
        total_tokens += len(tokenizer(r['prompt'], add_special_tokens=False)['input_ids']) - 1
    from tqdm import tqdm
    pbar = tqdm(total=max(total_tokens, 1), desc="tokens")
    with torch.no_grad():
        for request in tqdm(requests):
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            enc = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            enc = {k: v.to(model.device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
            L = input_ids.size(1)

            if args.quest:
                # Quest 分支：一次前向，别逐 token
                outputs = model(**enc, use_cache=False)
                logits = outputs.logits.log_softmax(dim=-1)
                pbar.update(max(L - 1, 0))
                values, indices = logits.squeeze(0).topk(dim=-1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
                gold_indices = input_ids[:, 1:]
                logprobs = [None] + torch.gather(logits, -1, gold_indices.unsqueeze(-1)).squeeze(-1).squeeze(0).detach().cpu().tolist()
                top_logprobs = [None] + [{tokenizer.convert_ids_to_tokens(i.item()): v.item()} for v, i in zip(values.squeeze(-1), indices.squeeze(-1))]
                result['result'] = {
                    "choices": [{
                        "text": prompt,
                        "logprobs": {"tokens": tokens, "token_logprobs": logprobs, "top_logprobs": top_logprobs, "text_offset": []},
                        "finish_reason": "length"
                    }],
                    "request_time": {"batch_time": 0, "batch_size": 1}
                }
                results.append(result)
                # 重置 Quest 状态，避免串样本
                try:
                    model.quest_manager.reset()
                except Exception:
                    try:
                        model.model.quest_manager.reset()
                    except Exception:
                        pass
                continue
            else:
                # 原整段前馈路径`
                logits = model(**enc, use_cache=False).logits.log_softmax(dim=-1)
                pbar.update(max(L-1, 0))
                values, indices = logits.squeeze(0).topk(dim=-1, k=1)
                tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
                gold_indices = input_ids[:, 1:]
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
            
            # --- 重要逻辑修正 ---
            # 原始代码在这里重置了 previous_hidden_states，但对于 UnionCachePolicy 来说，
            # 需要重置的是 cache_policy 自身的状态，以便处理下一个独立的 prompt。
            # 我们的 get_stats_and_reset 已经包含了清空内部 list 的功能，
            # 但还需要重置 UnionCachePolicy 的主缓存 self.cached_mask。
            if args.ours or args.enable_heavy_hitter_masker:
                if args.model_type == "opt":
                    layers_to_reset = model.model.decoder.layers
                else: # 假设 Llama, Qwen 等
                    layers_to_reset = model.model.layers
                
                for layer in layers_to_reset:
                    # 重置 previous_hidden_states (如果其他逻辑依赖它)
                    layer.self_attn.previous_hidden_states = None
                    # 关键：重置缓存策略的状态
                    if hasattr(layer.self_attn, 'cache_policy') and hasattr(layer.self_attn.cache_policy, 'reset'):
                        layer.self_attn.cache_policy.reset()
            if args.quest and hasattr(model, 'quest_manager') and model.quest_manager is not None:
                # 假设 quest_manager 实例挂载在 LlamaModel (self.model) 下，
                # 或者 LlamaForCausalLM (self) 下。
                # 访问路径可能需要根据你的具体实现调整，例如 model.model.quest_manager
                try:
                    # 路径1：如果 quest_manager 在 LlamaForCausalLM 级别
                    model.quest_manager.reset()
                except AttributeError:
                    try:
                        # 路径2：如果 quest_manager 在 LlamaModel 级别
                        model.model.quest_manager.reset()
                    except AttributeError:
                        print("Warning: Could not find quest_manager to reset.")
    if all_requests_stats:
        # 确定稀疏层的数量
        num_sparse_layers = len(all_requests_stats[0]) if all_requests_stats else 0
        
        if num_sparse_layers > 0:
            # 获取稀疏计算开始的层索引
            if args.model_type == "opt":
                full_prefix_layers = model.model.decoder.layers[0].self_attn.full_prefix_layers
            else:
                # 假设 Llama, Qwen 等模型的结构
                try:
                    full_prefix_layers = model.model.layers[0].self_attn.full_prefix_layers
                except AttributeError:
                    # 如果模型没有 full_prefix_layers, 默认从0开始
                    print("Warning: 'full_prefix_layers' not found. Assuming it is 0.")
                    full_prefix_layers = 0

            # 1. 准备用于保存的统计数据字典
            cache_stats_summary = {
                "per_layer_average_hit_rate": {}
            }
            avg_hit_rate_per_layer = [0.0] * num_sparse_layers

            print("\n--- Cache Hit Rate Statistics ---")
            for layer_idx in range(num_sparse_layers):
                total_hits_layer = sum(req[layer_idx]['total_hits'] for req in all_requests_stats)
                total_selections_layer = sum(req[layer_idx]['total_selections'] for req in all_requests_stats)
                
                layer_avg_hit_rate = (total_hits_layer / total_selections_layer) if total_selections_layer > 0 else 0.0
                avg_hit_rate_per_layer[layer_idx] = layer_avg_hit_rate
                
                actual_layer_index = layer_idx + full_prefix_layers
                print(f"Layer {actual_layer_index:2d}: Average Hit Rate = {layer_avg_hit_rate:.4f} ({total_hits_layer} / {total_selections_layer})")

                # 填充字典
                cache_stats_summary["per_layer_average_hit_rate"][f"layer_{actual_layer_index}"] = {
                    "hit_rate": layer_avg_hit_rate,
                    "total_hits": total_hits_layer,
                    "total_selections": total_selections_layer,
                }

            overall_avg = sum(avg_hit_rate_per_layer) / num_sparse_layers if num_sparse_layers > 0 else 0.0
            cache_stats_summary["overall_average_hit_rate"] = overall_avg
            
            print("-" * 35)
            print(f"Overall Average Hit Rate (across sparse layers): {overall_avg:.4f}")
            print("-" * 35)
            
            # 2. 将统计字典写入 JSON 文件
            # 文件名与主输出文件关联，例如 'result.jsonl' -> 'result.jsonl.stats.json'
            stats_output_path = f"{args.output_path}.stats.json"
            try:
                with open(stats_output_path, 'w') as f_stats:
                    json.dump(cache_stats_summary, f_stats, indent=4)
                print(f"Cache hit rate statistics have been saved to: {stats_output_path}")
            except Exception as e:
                print(f"Error: Failed to save cache stats to {stats_output_path}: {e}")

    # === 修改 END ===

    # if args.ours:
    #     density = sum(density) / len(density) * 100
    #     retain_ratio = (1 - math.sqrt(1 - density / 100)) * 100
    #     #print("\ndensity: %.2f"%(density))
    #     print("retain ratio: %.2f\n"%(retain_ratio))

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
