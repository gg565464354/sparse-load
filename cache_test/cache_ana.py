import torch
import sys
import time
import json
import matplotlib.pyplot as plt


def plot_multiple_lines(data, tags, x_label="Head ID", y_label="Hit Rate", title="Cache hit rate", fig_path="./tmp.png"):
    """
    将多个长度相同的列表绘制为折线图，并标注不同的标签。
    
    参数:
        data (list of list): 包含多个列表的数据，每个列表代表一条折线的 Y 值。
        tags (list of str): 每条折线对应的标签。
        x_label (str): X 轴名称。
        y_label (str): Y 轴名称。
        title (str): 图表标题。
    """
    # 确保数据和标签数量一致
    if len(data) != len(tags):
        raise ValueError("数据列表的数量必须与标签数量一致！")
    
    # 确保所有列表长度相同
    list_length = len(data[0])
    if not all(len(lst) == list_length for lst in data):
        raise ValueError("所有列表的长度必须相同！")
    
    # 生成 X 轴数据
    x = range(list_length)
    
    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图表大小
    for i, (y_values, tag) in enumerate(zip(data, tags)):
        plt.plot(x, y_values, label=tag, marker='o')  # 绘制折线并添加标签
    
    # 添加图例、标题和坐标轴标签
    plt.legend()  # 显示图例
    plt.title(title)  # 设置标题
    plt.xlabel(x_label)  # 设置 X 轴名称
    plt.ylabel(y_label)  # 设置 Y 轴名称
    
    # 显示网格
    plt.grid(True)
    
    # 显示图表
    plt.savefig(fig_path)

def get_hit_rate(a, b):
    # 统计 b 中每个元素在 a 中出现的次数
    set_a = set(a)
    cnt = 0
    cnt = sum(1 for bb in b if bb in set_a)

    # 占 b 所有元素的比例
    ratio = cnt / len(b)

    return ratio

torch.cuda.init()

N = 7800
N_p = 7800//2
d = 128
bh = 64
# bh = 64

idx_shape = (N_p, 1, bh)
kv_shape = (N, bh, d)
cache_shape = (N_p, bh, d)


layer_id = 0
print(f"####################### layer #{layer_id}")

data_name = f"./tmp/test2_layer{layer_id}.pt"
layer_idx = torch.load(data_name)
# decode_idxs = process_idx(layer_idx) 

print("layer_idx shape = ", layer_idx.shape)

token_num = layer_idx.shape[0]
head_num = layer_idx.shape[1]
topk = layer_idx.shape[2]

print(f"token_num={token_num}, head_num={head_num}, topk={topk}")

K = 4

head_hit_rates = [] # 异步更新的命中率
head_up_rate = [] # 更新方法的命中率上限

for h in range(head_num):
    tmp_list = []
    up_list = []
    for i in range(1, token_num):
        # 定期更新
        cache_id = i - i%K
        # 每次更新前还是用上一次的token
        if i%K == 0 and i != 0:
            cache_id -= K

        tmp_rate = get_hit_rate(layer_idx[cache_id][h].tolist(), layer_idx[i][h].tolist())
        tmp_rate_up = get_hit_rate(layer_idx[i-1][h].tolist(), layer_idx[i][h].tolist())
        # print("tmp_rate = ", tmp_rate)
        up_list.append(tmp_rate_up)
        tmp_list.append(tmp_rate)

        # print("tmp_list = ", tmp_list)

    head_hit_rates.append(tmp_list)
    head_up_rate.append(up_list)


K = 2
head_k2_rates = [] # 更新方法的命中率上限
for h in range(head_num):
    tmp_list = []
    for i in range(1, token_num):
        # 定期更新
        cache_id = i - i%K
        # 每次更新前还是用上一次的token
        if i%K == 0 and i != 0:
            cache_id -= K

        tmp_rate = get_hit_rate(layer_idx[cache_id][h].tolist(), layer_idx[i][h].tolist())
        # print("tmp_rate = ", tmp_rate)
        tmp_list.append(tmp_rate)

        # print("tmp_list = ", tmp_list)

    head_k2_rates.append(tmp_list)

# for h in range(head_num):
#     print(f"\n#{h} = {head_hit_rates[h]}")


K = 2
head_ada_rates = [] # 更新方法的命中率上限

################################## ada update
avg_hits = [0.9571017871017874, 0.9445117845117847, 0.9022662522662527, 0.8722817922817924, 0.9166692566692566, 0.9442553742553744, 0.8885340585340589, 0.8689484589484594, 0.8024864024864025, 0.9411137011137013, 0.9639419839419835, 0.8635146335146334, 0.9191012691012689, 0.8097358197358195, 0.8472856772856772, 0.8306086506086509, 0.9285884485884489, 0.7798083398083397, 0.8658974358974362, 0.892141932141932, 0.8686946386946387, 0.8235638435638436, 0.8765811965811968, 0.8903030303030305, 0.96001554001554, 0.9148433048433049, 0.8398938098938099, 0.7375110075110075, 0.8345066045066045, 0.8347422947422947, 0.8741491841491841, 0.8510852110852111, 0.8449313649313649, 0.8513105413105413, 0.8584900284900283, 0.943105413105413, 0.8764180264180261, 0.9705205905205908, 0.8979823879823876, 0.8490883190883188, 0.9362548562548562, 0.8775110075110075, 0.924680134680135, 0.8618207718207717, 0.7054364154364158, 0.8110178710178713, 0.8577130277130275, 0.8749210049210053, 0.8416731416731419, 0.8335275835275833, 0.8786428386428389, 0.9025511525511526, 0.7798886298886301, 0.9073970473970474, 0.9449028749028754, 0.8741103341103342, 0.7871691271691273, 0.9587412587412589, 0.8825174825174826, 0.9667573167573168, 0.8537321937321936, 0.9129629629629629, 0.8162289562289559, 0.8969360269360268]
# should_update = False
cur_cached_id = 0
head_ada_rates = [] # 更新方法的命中率上限
head_ada_update_num = []
for h in range(head_num):
    tmp_update = 0 
    tmp_list = []
    for i in range(1, token_num):
        tmp_rate = get_hit_rate(layer_idx[cur_cached_id][h].tolist(), layer_idx[i][h].tolist())

        if (1-tmp_rate) > ((1-avg_hits[h])*1.3):
            tmp_update += 1
            cur_cached_id = i

        tmp_list.append(tmp_rate)

    head_ada_update_num.append(tmp_update)
    head_ada_rates.append(tmp_list)





################################## show result
avg_rate_k4 = []
avg_rate_k2 = []
avg_rate_up = []
avg_rate_ada = []

for h in range(head_num):
    tmp_list = head_hit_rates[h]
    tmp_up = head_up_rate[h]
    tmp_k2 = head_k2_rates[h]
    tmp_ada = head_ada_rates[h]

    avg_k4 = sum(tmp_list)/len(tmp_list)
    avg_k2 = sum(tmp_k2)/len(tmp_k2)
    avg_up = sum(tmp_up)/len(tmp_up)
    avg_ada = sum(tmp_ada)/len(tmp_ada)

    k2_num = token_num//2
    k4_num = token_num//4
    up_num = token_num
    ada_num = head_ada_update_num[h]

    # print(f"\n################ head {h}")
    # print(f"K=4 {avg_k4} ({k4_num}) = {tmp_list}")
    # print(f"K=2 {avg_k2} ({k2_num})= {tmp_k2}")
    # print(f"K=1 {avg_up} ({up_num})= {tmp_up}")
    # print(f"Ada {avg_ada} ({ada_num})= {tmp_ada}")

    avg_rate_k4.append(avg_k4)
    avg_rate_k2.append(avg_k2)
    avg_rate_up.append(avg_up)
    avg_rate_ada.append(avg_ada)




#################################### 画图
for h in range(head_num):
    plot_multiple_lines(
        data=[head_up_rate[h], head_hit_rates[h]],
        tags=[f"{h}-up", f"{h}-k4"],
        fig_path=f"./fig/head{h}_k4_fig.png"
    )


# plot_multiple_lines(
#     data=[avg_rate_k4, avg_rate_k2, avg_rate_up, avg_rate_ada],
#     tags=['k4', 'k2', 'up', 'ada'],
#     fig_path=f"./head{h}_fig.png"
# )



# show avg hit rate
# avg_hits = []
# for h in range(head_num):
#     tmp_up = head_up_rate[h]

#     avg_up = sum(tmp_up)/len(tmp_up)

#     avg_hits.append(avg_up)

# print(f"avg_hits = {avg_hits}")


'''
    低命中率的head会导致整体通信开销大，完整放置到显存中可以保证加载性能
    高命中率的head显存利用率高，如果只缓存cache部分，缓存这部分是有收益的
    
'''