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
    parser.add_argument('--quest', action='store_true')  # 你代码里用了但没定义，补上
    parser.add_argument('--quest_recent_ratio', type=float, default=0.0)  # 例如 0.1 表示保留最近 10%
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
    args = parser.parse_args()
    if args.enable_heavy_hitter_masker:
        set_symlink(args.model_type, f"modeling_{args.model_type}_test.py")
    elif args.ours:
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
    elif args.quest:
        set_symlink(args.model_type, f"modeling_{args.model_type}_quest_query_merge_recent.py")
        # set_symlink(args.model_type, f"modeling_{args.model_type}_quest_query_merge.py")
    else:
        set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")


    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    # import sys
    # sys.path.insert(0,'/workspace/playground/libs')
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    config.quest_recent_ratio = args.quest_recent_ratio
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
    if args.model_path is None:
        # if args.model_type == 'opt' or args.model_type == 'llama':
        if args.model_type == 'opt' or args.quest:
        # if args.model_type == 'opt' :
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16, attn_implementation="eager", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

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
            attn.heavy_hitter_masker.heavy_budget_ratio = args.heavy_budget_ratio
            attn.heavy_hitter_masker.recent_budget_ratio = args.recent_budget_ratio

    model.half().eval().cuda()

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

results = []
density = []
# 用来记录最后一个 token 是否预测正确
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for request in tqdm.tqdm(requests):
        result = {'request': request, 'result': {}}
        prompt = request['prompt']
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

        # 如果序列长度不足以进行拆分，则跳过
        if input_ids.shape[1] <= 21:
            continue

        # 1. 拆分输入序列
        # 将输入序列拆分为 prefill 部分和 decode 部分（最后 20 个 token）
        prefill_input_ids = input_ids[:, :-20]
        decode_input_ids = input_ids[:, -20:]

        # 2. Prefill 阶段
        # 一次性处理 prefill 部分，生成初始的 KV cache
        # `use_cache=True` 会让模型返回 past_key_values
        outputs = model(prefill_input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # 3. Decode 阶段 (逐个 token 模拟)
        # 循环处理最后 20 个 token 中的前 19 个，以更新 KV cache
        # 这一步是关键，它会持续调用你的自定义 attention 逻辑
        for i in range(19):
            # 每次只输入一个 token，并传入上一轮的 KV cache
            current_token_id = decode_input_ids[:, i:i+1]
            outputs = model(current_token_id, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values

        # 4. 获取最终预测
        # 输入倒数第二个 token (即 decode_ids 的第 19 个)，得到对最后一个 token 的预测 logits
        last_step_input_id = decode_input_ids[:, -2:]
        last_step_input_id = last_step_input_id[:,:1]
        
        outputs = model(last_step_input_id, use_cache=True, past_key_values=past_key_values)
        
        # logits 的形状是 (batch_size, sequence_length, vocab_size)
        # 在这里，sequence_length 是 1，所以我们取 [:, -1, :]
        final_logits = outputs.logits[:, -1, :]

        # 真实的目标 token 是原始序列的最后一个 token
        target_token_id = decode_input_ids[:, -1]

        # 检查预测是否正确
        predicted_token_id = torch.argmax(final_logits, dim=-1)
        if predicted_token_id == target_token_id:
            correct_predictions += 1
        total_predictions += 1

        # (可选) 如果你仍然需要原始的输出格式，可以像下面这样构建
        # 注意：这里的 logprobs 只反映了最后一个 token 的情况，不再是整个序列
        log_probs_full = final_logits.log_softmax(dim=-1)
        target_log_prob = torch.gather(log_probs_full, -1, target_token_id.unsqueeze(-1)).item()
        
        # (这里简化了 result 的填充，因为原始的 per-token logprob 不再适用)
        result['result'] = {
            "choices": [
                {
                    "text": f"Prediction for last token. Correct: {predicted_token_id.item() == target_token_id.item()}",
                    "logprobs_for_last_token": target_log_prob
                }
            ],
        }
        results.append(result)

        # 重置状态，为下一个请求做准备
        qm = getattr(getattr(model, "model", None), "quest_manager", None)
        if qm is not None:
            qm.reset()
        if args.ours:
            if args.model_type in ["opt", "llama", "qwen2", "qwen3"]:
                layers = model.model.decoder.layers if args.model_type == "opt" else model.model.layers
                for layer in layers:
                    if hasattr(layer.self_attn, 'previous_hidden_states'):
                        layer.self_attn.previous_hidden_states = None

# 在循环结束后打印最终的准确率
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n[Final Result] Accuracy for the last token prediction: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

# ==============================================================================
#  替换你原始代码中的 with open(output_path, 'w') as f: 块
# ==============================================================================
with open(output_path, 'w') as f:
    # 写入一个总结性的结果
    summary = {
        "last_token_accuracy": (correct_predictions / total_predictions) if total_predictions > 0 else 0,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }
    f.write(json.dumps(summary) + '\n\n')
    # 写入详细的 per-request 结果
    for result in results:
        f.write(json.dumps(result) + '\n')