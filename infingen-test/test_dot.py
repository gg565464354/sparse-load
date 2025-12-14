import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer

# 配置参数
b = 1
n = 4000          # 序列长度
d_model = 4096    # 模型维度
d_ff = 4 * d_model  # MLP隐藏层维度
num_heads = 32    # 注意力头数
head_dim = d_model // num_heads  # 每头的维度
num_runs = 10    # 每个模块的运行次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义Q/K/V投影层（输出维度为d_model，按头分割）
class QKVProjection(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        q = self.W_q(x)  # [n, d_model]
        q = self.W_k(x)  # [n, d_model]
        q = self.W_v(x)  # [n, d_model]
        return q

# 定义多头注意力计算
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
    
    def forward(self, q, k, v):
        # 分头操作：[n, d_model] -> [n, num_heads, head_dim] -> [num_heads, n, head_dim]
        q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        
        # 计算注意力分数 [num_heads, n, n]
        attn_scores = torch.matmul(q, q.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 注意力加权并合并头 [num_heads, n, head_dim] -> [n, d_model]
        output = torch.matmul(attn_probs, q)  # [num_heads, n, head_dim]
        output = output.transpose(0, 1).contiguous().view(n, self.d_model)
        return output

# 定义多头注意力计算
class MultiHeadFlashAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
    
    def forward(self, q, k, v):
        # 分头操作：[n, d_model] -> [n, num_heads, head_dim] -> [num_heads, n, head_dim]
        q = q.view(b, self.num_heads, n, self.head_dim)
        k = k.view(b, self.num_heads, n, self.head_dim)
        v = v.view(b, self.num_heads, n, self.head_dim)
        
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            output = F.scaled_dot_product_attention(q, k, v)
        return output

# 定义MLP（与之前相同）
class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# 初始化输入数据
x = torch.randn(b, n, d_model, device=device)  # [n, d_model]

# 初始化模块
qkv_proj = QKVProjection(d_model, num_heads).to(device)
multihead_attn = MultiHeadAttention(d_model, num_heads).to(device)
flash_attn = MultiHeadFlashAttention(d_model, num_heads).to(device)
mlp = MLP(d_model, d_ff).to(device)

# 预热GPU
with torch.no_grad():
    _ = qkv_proj(x)
    #_ = multihead_attn(x, x, x)
    _ = flash_attn(x, x, x)
    _ = mlp(x)

# 计时函数
def benchmark(module, inputs, num_runs=100):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = module(*inputs if isinstance(inputs, tuple) else inputs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs  # 毫秒/次

# 分别计时
qkv_time = benchmark(qkv_proj, x, num_runs)
#attn_time = benchmark(multihead_attn, (x, x, x), num_runs)
flash_attn_time = benchmark(flash_attn, (x, x, x), num_runs)
x = x.view(b, n, d_model)
mlp_time = benchmark(mlp, x, num_runs)

# 输出结果
#total_time = qkv_time + attn_time + mlp_time
total_time2 = qkv_time + flash_attn_time + mlp_time
print(f"序列长度 n={n}, 模型维度 d_model={d_model}, 头数 num_heads={num_heads}")
print(f"Q/K/V投影时间: {qkv_time:.3f} ms | 占比: {qkv_time / total_time2 * 100:.1f}%")
#print(f"多头注意力时间: {attn_time:.3f} ms | 占比: {attn_time / total_time * 100:.1f}%")
print(f"Flash attention时间: {flash_attn_time:.3f} ms | 占比: {flash_attn_time / total_time2 * 100:.1f}%")
print(f"MLP计算时间: {mlp_time:.3f} ms | 占比: {mlp_time / total_time2 * 100:.1f}%")