import torch
import torch.nn.functional as F
import my_cache_load._C as _C
from concurrent.futures import ThreadPoolExecutor

class CacheManager:
    def __init__(self):
        """
        初始化 CacheManager，使用字典保存不同 layer_id 对应的缓存数据。
        """
        self._caches = {}  # key: layer_id, value: cache data
        self._basic_group_head_ids = {} # key: layer_id, value: layer group head ids
        self.layer_head_numbers = 64
        self.executor = ThreadPoolExecutor()  # 可复用线程池

    def init_basic_group_head_ids(self, layer_group_head_ids):
        for l in range(len(layer_group_head_ids)):
            self._basic_group_head_ids[l] = layer_group_head_ids[l]

    def add_cache(self, layer_id, batch_size, head_num, sparse_len, hidden_size):
        """
        添加或更新指定 layer_id 的缓存数据。
        
        :param layer_id: 缓存标识符（字符串）
        :param data: 要存储的数据（任意类型）
        """
        if layer_id in self._caches:
            raise ValueError("Python ERROR! [add_cache] layer id is used!")
            return 1

        new_cache = self.create_cache_instance(batch_size, head_num, sparse_len, hidden_size)

        self._caches[layer_id] = new_cache
        return 0

    def remove_cache(self, layer_id):
        """
        移除指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        """
        if layer_id in self._caches:
            del self._caches[layer_id]

    def has_cache(self, layer_id) -> bool:
        """
        检查是否存在指定 layer_id 的缓存。
        
        :param layer_id: 缓存标识符
        :return: 布尔值
        """
        return layer_id in self._caches

    def clear_caches(self):
        """
        清空所有缓存。
        """
        self._caches.clear()

    def cache_count(self) -> int:
        """
        返回当前缓存的数量。
        
        :return: 整数
        """
        return len(self._caches)


    def cpu_cache_load_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        cur_cache = self.cache[layer_id]
        group_cached_k, group_cached_v = cur_cache.get_cached_kv()

        group_final_k = []
        group_final_v = []

        # Step 1: 提前异步提交 CPU 计算任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_unhit = executor.submit(cur_cache.get_unhit_kv, prefetch_idx, all_k, all_v)

            # Step 2: 同时开始 cached_k/v 的 GPU 传输（异步）
            with torch.cuda.stream(transfer_stream):
                group_cached_gpu_k = [k.cuda(non_blocking=True) for k in group_cached_k]
                group_cached_gpu_v = [v.cuda(non_blocking=True) for v in group_cached_v]

            # Step 3: 等待 CPU 计算结束
            group_unhit_k, group_unhit_v, group_unhit = future_unhit.result()

        # Step 4: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)

        transfer_stream.synchronize()

        return group_final_k, group_final_v, group_unhit


    def gpu_cache_load_asyn(self, transfer_stream, prefetch_idx, all_k, all_v, group_cached_gpu_k, group_cached_gpu_v):
        cur_cache = self.cache[layer_id]

        group_final_k = []
        group_final_v = []

        # Step 1: 提前异步提交 CPU 计算任务
        group_unhit_k, group_unhit_v, group_unhit = cur_cache.get_unhit_kv(prefetch_idx, all_k, all_v)

        # Step 2: unhit_k/v -> GPU 传输 + 拼接（仍在 transfer_stream）
        with torch.cuda.stream(transfer_stream):
            unhit_gpu_k = [k.cuda(non_blocking=True) for k in group_unhit_k]
            unhit_gpu_v = [v.cuda(non_blocking=True) for v in group_unhit_v]

            for i in range(len(group_cached_gpu_k)):
                final_k = torch.cat([group_cached_gpu_k[i], unhit_gpu_k[i]], dim=0)
                final_v = torch.cat([group_cached_gpu_v[i], unhit_gpu_v[i]], dim=0)
                group_final_k.append(final_k)
                group_final_v.append(final_v)

        transfer_stream.synchronize()

        return group_final_k, group_final_v, group_unhit

    
    def load_and_update_cache(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v = _C.generate_cache(cur_idx_list[i], all_k, all_v)

        stat = self._caches[layer_id].update_group_cache(cur_idx_list[i], group_cpu_k, group_cpu_v)
        if stat != 0:
            raise ValueError("Python ERROR! [load_and_update_cache] Update cache Fail!")

        return group_gpu_k, group_gpu_v, group_cpu_k, group_cpu_v
    
    async def update_cpu_cache_asyn(self, layer_id, transfer_stream, prefetch_idx, all_k, all_v):
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            self.executor,
            self._cache[layer_id].asyn_update_cache,
            prefetch_idx,
            keys,
            values
        )

        return result


    def generate_class_group_ids(self, layer_id, batch_size):
        """
        Require static offline head group
        """
        cur_basic_head_ids = self._basic_group_head_ids[layer_id]
        cur_head_num = self.layer_head_numbers
        final_head_ids = [[] for g in range(len(cur_basic_head_ids))]
        
        for b in range(len(batch_size)): # 每个batch的head id 依次添加
            for g in range(len(cur_basic_head_ids)): # 遍历每个group
                for head_id in cur_basic_head_ids[g]: # 添加所有的head id，最大值为 bh*cur_head_num
                    final_head_ids[g].append(head_id + b*cur_head_num)

        return final_head_ids


    # 初始化c++模块中的cache
    def create_cache_instance(self, batch_size, head_num, sparse_len, hidden_size):
        """ Initial the cache
        Call for every new batch of requests
        Building cache for every decoder layer.
        """
        if len(self._basic_group_head_ids == 0):
            raise ValueError("Python ERROR! [create_cache_instance] basic_group_ids was not initialed!")

        # 构造 class group id
        new_class_group_id = generate_class_group_ids(batch_size, head_num)
        
        cache_shape = (batch_size*head_num, sparse_len, hidden_size)

        # 初始化一个cache
        cache = _C.CPUCache(bh, cache_idx, cache_shape, new_class_group_id)
        
        return cache

        


##################################### 传统方法

def select_kv(prefetch_idx, k_cache, v_cache):
    """Selects and aggregates critical KV caches using speculated indices

    On the decoding stage, aggregates the critical KV caches corresponding to
    the speculated prefetch index using embedding function.

    Args:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
        k_cache: Key cache (n, bh, d)
        v_cache: Value cache (n, bh, d)

    Returns:
        selected_k: selected key cache (n', bh, d)
        selected_v: selected value cache (n', bh, d)
    """

    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    return selected_k, selected_v


def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """Speculates the indices of the critical KV caches of next attention layer.

    On the decoding stage, by using the hidden states (layer i), partial query
    weight (layer i+1), and partial key cache (layer i+1), speculates the
    attention score of the next layer. After that, counts the number of
    critical tokens and gets the indcies of the top-k KV cache tokens with high
    attention scores.

    Args:
        hidden: Hidden states of layer i (b, 1, D)
        p_w_q: Partial query weight (D', D)
        p_k_c: Partial key cache (n, bh, d')

        Note that bh * d' == D'

    Returns:
        prefetch_idx: Indices of critical KV cache tokens for each head and batch (n', 1, bh)
    """
    b = hidden.shape[0]
    p_q = F.linear(hidden, p_w_q, bias=None)
    p_q = p_q.view(b, 1, n_head, -1)
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    max_ = torch.max(p_attn, dim=-1)[0]
    thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    count = torch.where(
        p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    )
    mean = torch.mean(torch.sum(count, dim=-1)).item()
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    )[1]

    return prefetch_idx