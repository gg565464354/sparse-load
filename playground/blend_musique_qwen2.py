import os, sys, json
import numpy as np
import torch
local_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
sys.path.insert(0, local_lib_path)
# 确保优先使用本地 transformers 而不是 site-packages
if 'transformers' in sys.modules:
    del sys.modules['transformers']

# 临时禁用 torchvision 以避免版本冲突
sys.modules['torchvision'] = None
sys.modules['torchvision.transforms'] = None

print(f"Using transformers from: {__import__('transformers').__file__}")
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from utils import load_dataset, normalize_question, build_qa_prompt, compute_f1
USE_HF = True  # 关键：使用 HF 路径以启用自定义 modeling_qwen2.py

eval_dataset = load_dataset("/workspace/CacheBlend/inputs/musique_s.json")
model_name_or_path = "/root/model/Qwen2-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

if USE_HF:
    model = Qwen2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # 注入 skewing_matrix 与 partial_weight_q（参考 run_lm_eval_harness.py 的 qwen2 分支）
    try:
        skew_path = "/workspace/SparseCache/accuracy/setup/skewing_matrix_qwen2/Qwen2-1.5B-Instruct.pt"
        weights_dir = "/workspace/SparseCache/accuracy/setup/weights_qwen2/Qwen2-1.5B-Instruct_0.2"
        A = torch.load(skew_path, map_location="cpu")

        partial_weight_ratio = 0.2
        alpha = 5.0
        capacity = 1.0
        budget = 0.2

        device = model.device
        dtype = next(model.parameters()).dtype

        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            attn.partial_weight_ratio = partial_weight_ratio
            attn.partial_weight_q = torch.load(
                os.path.join(weights_dir, f"partial_weight_q_{layer_idx}.pt"),
                map_location="cpu"
            ).to(device=device, dtype=dtype)
            attn.alpha = alpha
            attn.capacity = capacity
            attn.budget = budget
            skew = A[layer_idx] if isinstance(A, (list, tuple)) else A[layer_idx]
            attn.skewing_matrix = skew.to(device=device, dtype=dtype)
    except Exception as e:
        print(f"Warning: failed to inject weights/skewing_matrix: {e}")
else:
# vLLM 路径（不会使用自定义 modeling_qwen2.py 的逻辑，仅保留以便需要 TTFT）
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name_or_path, gpu_memory_utilization=0.3)
    llm.set_tokenizer(tokenizer)

prefix_prompt = "You will be asked a question after reading several passages. Please directly answer the question based on the given passages. Do NOT repeat the question. The answer should be within 5 words..\nPassages:\n"
query_prompt = "\n\nAnswer the question directly based on the given passages. Do NOT repeat the question. The answer should be within 5 words. \nQuestion:"

ttft_full = []
f1_full = []

for ex in eval_dataset:
    answers = ex["answers"]
    doc_prompts, q_prompt = build_qa_prompt(ex, query_prompt)

    s_start_full_text = "<|im_start|>system\n" + prefix_prompt + "<|im_end|>\n<|im_start|>user\n"
    s_end_text = "<|im_end|>\n<|im_start|>assistant\n"

    input_prompt = (
        (tokenizer.bos_token or "")
        + s_start_full_text
        + "".join(doc_prompts)
        + q_prompt
        + s_end_text
    )

    if USE_HF:
        inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
                # 显式重置采样参数为默认值以避免警告
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
        gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
        res = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # 清理 ours 相关的缓存（如实现了该属性）
        try:
            for layer in model.model.layers:
                if hasattr(layer.self_attn, "previous_hidden_states"):
                    layer.self_attn.previous_hidden_states = None
        except Exception:
            pass
        ttft = None  # HF 路径不统计 vLLM 风格 TTFT
    else:
        sampling_params = SamplingParams(temperature=0, max_tokens=32)
        output = llm.generate([input_prompt], sampling_params)
        res = output[0].outputs[0].text
        ttft = output[0].metrics.first_token_time - output[0].metrics.first_scheduled_time
        print(f"TTFT with full prefill: {ttft}")
        ttft_full.append(ttft)

    print(f"Normal generation: {res}")
    f1 = max([compute_f1(res, answer, tokenizer) for answer in answers])
    f1_full.append(f1)
    print("------------")

print("---------------Result Summary---------------------")
if len(ttft_full):
    print(f"TTFT with full prefill: {np.mean(ttft_full)}")
print(f"F1 with full prefill: {np.mean(f1_full)}")
