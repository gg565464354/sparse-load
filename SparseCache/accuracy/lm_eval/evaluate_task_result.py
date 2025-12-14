import argparse
import json
import os

from lm_eval import evaluator, tasks
from tasks import EvalHarnessAdaptor

def json_to_key(obj):
    return json.dumps(obj)


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--result-file', type=str, default='result.jsonl')
    parser.add_argument('--task-name', type=str, default='hellaswag')
    parser.add_argument('--model-type', type=str, default='opt')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num-fewshot', type=int, default=0)
    args = parser.parse_args()
    
    if args.model_type == 'opt':
        os.environ['MODEL_NAME'] = "facebook/opt-66b"
    elif args.model_type == 'bloom':
        os.environ['MODEL_NAME'] = "bigscience/bloom"
    elif args.model_type == 'gpt_neox':
        os.environ['MODEL_NAME'] = "EleutherAI/gpt-neox-20b"
    elif args.model_type == 'llama':
        os.environ['MODEL_NAME'] = "/share/models/Llama-3.2-3B"
    elif args.model_type == 'qwen2':
        os.environ['MODEL_NAME'] = "/share/models/Qwen2-7B"
    elif args.model_type == 'qwen3':
        os.environ['MODEL_NAME'] = "/share/models/Qwen3-4B"
    else:
        assert False

    seq = 1024
    total_batch = 1
    pe = 'fixed'

    class RealRunner:
        
        def __init__(self, args):
            
            self.results = {}
            
            with open(args.result_file, 'r') as f:
                
                for line in f:
                    if line.strip() == '':
                        continue
                    
                    item = json.loads(line)
                    
                    request = item['request']
                    result = item['result']
                    
                    self.results[json_to_key(request)] = result
            
            print(f"{len(self.results)} items in the cache")
        
        def eval(self, batch):
            
            from tasks.eval_harness import tokenizer
            
            mask_loss = []
            each_correct = []

            for i, text in enumerate(batch['text']):
                
                request = {
                        "best_of": 1, 
                        "echo": True, 
                        "logprobs": 1, 
                        "max_tokens": 0, 
                        "model": "x", 
                        "n": 1, 
                        "prompt": text, 
                        "request_type": "language-model-inference", 
                        "stop": None, 
                        "temperature": 0, 
                        "top_p": 1
                    }
                
                key = json_to_key(request)
                
                correct = True
                
                if key in self.results:
                    result = self.results[key]
                    
                    token_logprobs = result['choices'][0]['logprobs']['token_logprobs']
                    tokens = result['choices'][0]['logprobs']['tokens']
                    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
                    assert token_logprobs[0] is None
                    
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    
                    obs = batch['obs'][i]
                    target = batch['target'][i]
                    eval_mask = batch['eval_mask'][i]
                    
                    n_positive = 0
                    sum_lobprob = 0
                    if args.debug:
                        print(target)
                    for i, mask in enumerate(eval_mask):
                        try:
                            
                            if i+1 >= len(tokens):
                                break
                            
                            if mask == True:
                                if args.debug:
                                    print(tokens[i+1], next(iter(top_logprobs[i+1].keys())))
                                correct = correct and (tokens[i+1] == next(iter(top_logprobs[i+1].keys())))
                                sum_lobprob += token_logprobs[i+1]
                                n_positive += 1
                        except Exception as e:
                            raise e
                    
                    # avg_logprob = sum(token_logprobs[1:]) / (len(token_logprobs) - 1)
                    # avg_logprob = sum_lobprob / n_positive
                    avg_logprob = (sum_lobprob / n_positive) if n_positive > 0 else 0.0
                    
                    mask_loss.append( - avg_logprob)
            
                    each_correct.append( correct )
                    
                else:
                    assert False
                

            out = {
                'mask_loss': mask_loss,
                'each_correct': each_correct,
            }
            
            
            return out

    t = RealRunner(args)

    adaptor = EvalHarnessAdaptor(t, seq, total_batch, shrink=pe != "fixed")

    results = evaluator.simple_evaluate(
        model=adaptor, 
        tasks=[args.task_name],
        num_fewshot=args.num_fewshot,
        limit=None
    )
    
    # 显式关闭多进程池
    if hasattr(adaptor, 'pool') and adaptor.pool is not None:
        try:
            adaptor.pool.close()
            adaptor.pool.join()
        except:
            pass
    
    dumped = json.dumps(results, indent=2)
    print(dumped)

    # === 新增 START: 读取并打印缓存命中率统计文件 ===
    # 基于主结果文件名构造统计文件的路径
    stats_file_path = f"{args.result_file}.stats.json"
    
    if os.path.exists(stats_file_path):
        print("\n" + "="*50)
        print("Cache Hit Rate Statistics (from the previous run)")
        print("="*50)
        try:
            with open(stats_file_path, 'r') as f_stats:
                stats_data = json.load(f_stats)
                
                # 格式化打印
                if "overall_average_hit_rate" in stats_data:
                    print(f"Overall Average Hit Rate: {stats_data['overall_average_hit_rate']:.4f}")
                
                if "per_layer_average_hit_rate" in stats_data:
                    print("\nDetails per Layer:")
                    for layer, data in stats_data["per_layer_average_hit_rate"].items():
                        print(f"  - {layer}: {data['hit_rate']:.4f} ({data['total_hits']} / {data['total_selections']})")
                print("="*50)

        except Exception as e:
            print(f"Warning: Could not read or parse the stats file '{stats_file_path}': {e}")
    # === 新增 END ===
    # 评测结束后删除结果文件
    try:
        if os.path.exists(args.result_file):
            os.remove(args.result_file)
            print(f"Removed result file: {args.result_file}")
    except Exception as e:
        print(f"Warning: failed to delete result file {args.result_file}: {e}")
