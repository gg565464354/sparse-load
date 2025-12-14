import os
import sys
# ------------------- ⬇️ 添加这 6 行代码 ⬇️ -------------------
# 告诉 Python 去哪里寻找 'infinigen' 包
# 我们假设它在 'SparseCache' 目录下
# sparse_cache_root = "/root/sparse-load/SparseCache"
# if sparse_cache_root not in sys.path:
#     # 插入到搜索路径的开头
#     sys.path.insert(0, sparse_cache_root) 
# ------------------- ⬆️ 添加这 6 行代码 ⬆️ -------------------
os.environ.setdefault("HF_DATASETS_ALLOW_LOCAL_SCRIPT", "1")
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

def reset_kv_state(model):
    """
    强制清空模型中每一层的自定义 KV 缓存状态。
    适配 HuggingFace 原生模型对象。
    """
    # 1. 寻找模型的 layers 列表 (兼容 Llama, Qwen, Mistral 等结构)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"): # 兼容一些旧架构
        layers = model.transformer.h
    else:
        return

    # 2. 遍历每一层，重置特定属性
    for layer in layers:
        # 这里是 Script B 中原本的逻辑：清空 compressed_k_cache
        if hasattr(layer.self_attn, "compressed_k_cache"):
            layer.self_attn.compressed_k_cache = None


def set_symlink(model_type, fname):
    # model_path = "../transformers/src/transformers/models/" + model_type
    model_path = "/root/sparse-load/playground/libs/transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("/root/sparse-load/SparseCache/accuracy/benchmark/source/" + fname)
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

def parse_args(cmd_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        default="longchat-v1.5-7b-32k",
        choices=[
            "llama-3-8b-inst-262k",
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "vicuna-v1.5-7b-16k",
            "mistral-7b-inst",
            "llama-3-8b-inst-64k",
            "llama-3-8b-inst-1048k",
            "llama-3.1-8b",
            "mistral-7b-instruct-v0.2-local",
            "llama-2-7b-inst-32k"
        ],
    )
    ap.add_argument("--name", type=str, default="default")
    ap.add_argument('--model-type', type=str, default='llama')
    ap.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    ap.add_argument("--arkvale", action="store_true", help="Enable ArkVale")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-budgets", type=int, default=128)
    ap.add_argument("--n-unlimited-layers", type=int, default=2)
    ap.add_argument("--n-max-bytes", type=int, default=40 * (1 << 30))
    ap.add_argument("--n-max-cpu-bytes", type=int, default=80 * (1 << 30))
    ap.add_argument("--page-topks", type=int, default=32)
    ap.add_argument("--n-win-pages", type=int, default=2)
    ap.add_argument("--n-sink-pages", type=int, default=1)
    ap.add_argument("--use-3-stages-gen", action="store_true")
    # InfiniGen
    ap.add_argument('--infinigen', action='store_true')
    ap.add_argument("--partial_weight_ratio", type=float, default=0.1)
    ap.add_argument("--partial_weight_path", type=str)
    ap.add_argument("--skewing_matrix_path", type=str)
    ap.add_argument("--alpha",type=float, default=5)
    ap.add_argument("--capacity",type=float, default=1.0)
    ap.add_argument("--budget",type=float, default=0.2)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
            "lcc",
            "repobench-p",
        ],
    )
    args = ap.parse_args(cmd_args)
    args.e = False
    if args.page_budgets < 0:
        args.page_budgets = None
    return args


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif ("llama2" in model_name) or ("llama-2" in model_name):
        # LLaMA-2 chat template with BOS and system prompt
        sys_prompt = (
            "You are a helpful, respectful, and honest assistant. Always answer as concisely as possible."
        )
        bos = getattr(tokenizer, "bos_token", None) or ""
        prompt = (
            f"{bos}[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        )
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    out_path,
    use_3_stages_gen=False,
):
    start = 0
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            start = len(list(f))
    if start >= len(data):
        return
    print(out_path, start)

    if use_3_stages_gen:
        from arkvale.adapter import generate

        generate.enable_3_stages_gen()
        delim = "1145141919810"
        prompt_format, q_prompt_format = prompt_format.split("{context}")
        prompt_format = prompt_format + "{context}" + delim + q_prompt_format

    data_ = data[start:]
    for json_obj in tqdm(data_):
        reset_kv_state(model)        # 1. 重置模型层内部状态
        torch.cuda.empty_cache()     # 2. 显式释放 PyTorch 显存碎片
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if use_3_stages_gen:
            prompt, q_prompt = prompt.split(delim)
            q_input_ids = tokenizer(q_prompt, truncation=False, return_tensors="pt").to(
                device
            )["input_ids"]
            generate.reset_q_input_ids(q_input_ids)
        if "chatglm3" in model_name:
            assert False
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(
                    device
                )
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        cur_max_gen = max_gen
        if use_3_stages_gen:
            context_length += q_input_ids.shape[-1]
            cur_max_gen += q_input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=cur_max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
                pad_token_id=tokenizer.eos_token_id,
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=cur_max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "length": json_obj["length"],
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        if use_3_stages_gen:
            generate.reset_q_input_ids()

    if use_3_stages_gen:
        generate.disable_3_stages_gen()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

import os
import json
# 导入 LlamaConfig，这是本次修复的关键
from transformers import AutoConfig, LlamaConfig 

def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(
        path, use_fast=False
    )
    # Ensure pad token exists (fallback to EOS) to avoid generation pathologies
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 手动读取 config.json 文件为一个 Python 字典 (保持不变)
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 2. 在字典层面直接修改 `rope_scaling` (保持不变)
    # if 'rope_scaling' in config_dict and isinstance(config_dict['rope_scaling'], dict) and config_dict['rope_scaling'].get('rope_type') == 'llama3':
    #     print("Patching rope_scaling in config dictionary before loading...")
    #     original_factor = config_dict['rope_scaling'].get('factor', 8.0)
    #     config_dict['rope_scaling'] = {"type": "linear", "factor": original_factor}

    # ==================== 核心修改点 ====================
    # 3. 不再使用 AutoConfig.from_dict，而是直接用 LlamaConfig 实例化
    #    `**config_dict` 会将字典 {'key1': 'val1', ...} 转换为
    #    LlamaConfig(key1='val1', ...) 这样的函数调用。
    #    这是最基本、兼容性最好的方法。
    # config = LlamaConfig(**config_dict)
    config = AutoConfig.from_pretrained(path)
    # ======================================================


    if args.arkvale:
        from arkvale import adapter

        model = AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            device_map=device,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        
        adapter.enable_arkvale(
            model, 
            dtype=torch.float16, 
            device=device, 
            page_size=32,
            page_budgets=4096 // 32,
            page_topks=32,
            n_max_bytes=5 * (1 << 30),
            n_max_cpu_bytes=60 * (1 << 30),
        )
    else:
        print(f"Loading {args.model_type} with Flash Attention 2...")
        # 强制指定使用 bfloat16，这是 Llama-3 + FA2 的最佳实践
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if args.model_type == "llama":
            model = AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation="flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                config=config,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="flash_attention_2"
            )
    model = model.eval()

    # Keep model pad_token_id consistent with tokenizer
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

# # ==================== 在这里插入调试代码 ====================
#     print("================= ARKVALE DEBUG INFO =================")
#     print(f"Model Config Class: {type(model.config)}")
#     print(f"Num Hidden Layers: {getattr(model.config, 'num_hidden_layers', 'Not Found')}")
#     print(f"Hidden Size: {getattr(model.config, 'hidden_size', 'Not Found')}")
#     print(f"Num Attention Heads (Q): {getattr(model.config, 'num_attention_heads', 'Not Found')}")
#     print(f"Num Attention Heads (KV): {getattr(model.config, 'num_key_value_heads', 'Not Found')}")
#     print(f"Head Dim: {model.config.hidden_size // model.config.num_attention_heads if hasattr(model.config, 'hidden_size') and hasattr(model.config, 'num_attention_heads') else 'Calc Error'}")
#     print("======================================================")
#     # ==========================================================

    return model, tokenizer


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("longbench_config/model2path.json", "r"))
    model2maxlen = json.load(open("longbench_config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    if args.infinigen:
        # Use ours only if at least one of the auxiliary assets exists
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
    else:
        set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")
    
    # Force reload the transformers.models.llama module after changing symlink
    import importlib
    if f'transformers.models.{args.model_type}' in sys.modules:
        print(f"Reloading transformers.models.{args.model_type} module...")
        # Remove all llama-related modules from cache
        modules_to_remove = [k for k in sys.modules.keys() if f'transformers.models.{args.model_type}' in k]
        for module_name in modules_to_remove:
            del sys.modules[module_name]
        print(f"Removed {len(modules_to_remove)} cached modules")
    
    max_length = model2maxlen[model_name]
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], model_name, device, args
    )
    if args.infinigen:
        if args.model_type == "opt":
            for layer in range(len(model.model.decoder.layers)):
                model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.decoder.layers[layer].self_attn.alpha = args.alpha
                model.model.decoder.layers[layer].self_attn.capacity = args.capacity
                model.model.decoder.layers[layer].self_attn.budget = args.budget
        if args.model_type == "llama":
            # Optional skewing_matrix
            A = None
            if args.skewing_matrix_path is not None and os.path.exists(args.skewing_matrix_path):
                A = torch.load(args.skewing_matrix_path)
            for layer in range(len(model.model.layers)):
                la = model.model.layers[layer].self_attn
                la.partial_weight_ratio = args.partial_weight_ratio
                # Optional partial_weight_q
                if args.partial_weight_path is not None and os.path.isdir(args.partial_weight_path):
                    pwq_file = os.path.join(args.partial_weight_path, f"partial_weight_q_{layer}.pt")
                    if os.path.exists(pwq_file):
                        la.partial_weight_q = torch.load(pwq_file)
                    else:
                        la.partial_weight_q = None
                else:
                    la.partial_weight_q = None
                la.alpha = args.alpha
                la.capacity = args.capacity
                la.budget = args.budget
                if A is not None:
                    la.skewing_matrix = A[layer]
    datasets = args.datasets
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("longbench_config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("longbench_config/dataset2maxlen.json", "r"))
    # predict on each dataset
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            os.makedirs(f"pred_e/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred_e/{model_name}/{args.name}/{dataset}.jsonl"
        else:
            # data_path = "/root/longbench_data/data" # <--- 指定解压后的 data 目录
            # data = load_dataset("THUDM/LongBench", dataset, split="test", data_dir=data_path) # <--- 添加 data_dir
            data = load_dataset("THUDM/LongBench", dataset, split="test")
            os.makedirs(f"pred/{model_name}/{args.name}", exist_ok=True)
            out_path = f"pred/{model_name}/{args.name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        get_pred(
            model,
            tokenizer,
            list(data),
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            out_path,
            use_3_stages_gen=args.use_3_stages_gen,
        )
