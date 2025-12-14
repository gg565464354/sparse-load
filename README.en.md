# SparseLoad

## Description
一个服务于LLM推理的稀疏offload框架

## 代码结构介绍

### 任务一 cache 数据分析与测试

1. 数据处理：`./cache_test/csrc/data_process.py` 将每个layer的关键token id数据（已经保存成单独的文件）处理成为一个统一的json文件
2. cache命中率模拟： `./cache_test/csrc/cache_ana.py` 模拟cache过程进行命中率测试


### 任务二 cache 功能实现（c++）

主要项目路径：`./csrc/my_cache_load`
项目结构特点：pytorch extension 格式的
安装方法: `python setup.py install`
安装结果: 可以使用 `import my_cache_load._C as _C` 来调用c++实现的 cache功能模块

c++ 项目文件介绍：
1. `./pybind_wrapper.cpp`: 声明了打包后的python 模块各个api对应的函数名和类型
2.  `./src/cpu_cache.cpp` 以及相应的 `.h`文件: 主要的cache功能实现模块，实现了非常多版本的cache，请直接关注最下面的版本

PS：当前最新的cache接口实现 `get_unhit_kv_tensor_v5`，功能是检索未命中KV，然后拼接未命中kv的成为新的tensor并进行padding

### 任务三 完整推理系统 （基于infinigen修改而成）

项目路径：`./MyCache`
主要工作路径：`./MyCache/speedup/flexgen/mycache`

重要文件：(位于主要工作路径下面)
1. `cache_selection_controller_v2.py` 实现了cache模块的主要功能定义
2. `flex_opt.py` 实现了各个layer定义以及各个计算，通信，存储操作的实现
3. `pytorch_backend.py`  实现了基础的Attention算子定义等内容，里面最重要的是 `patch_mha_gen()` 是我自己实现的attention算子

其他重要文件：
1. `./MyCache/speedup/infinigen/infinigen/kv_selection_controller.py`: 计算关键token的函数是调用的这里面的实现

执行方法：先下载好模型 `opt 6.7b`
1. 安装：执行 `./install.sh`，参考infinigen 本身的安装逻辑
2. 开始运行：`./MyCache/speedup/my_cache_bench.py`，基本逻辑是把我们实现的代码拷贝进工作区，然后输入参数进行执行
