# # 定制化开发环境, AI 通用版
# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# # 修改镜像源、设置时区、安装系统包和工具（合并RUN以减少层）
# RUN sed -i 's/archive.ubuntu.com/mirrors.cloud.tencent.com/g' /etc/apt/sources.list && \
#     sed -i 's/security.ubuntu.com/mirrors.cloud.tencent.com/g' /etc/apt/sources.list && \
#     sed -i 's/cn.archive.ubuntu.com/mirrors.cloud.tencent.com/g' /etc/apt/sources.list && \
#     ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
#     echo "Asia/Shanghai" > /etc/timezone && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#     git-lfs curl wget git axel nload libgl1-mesa-glx ffmpeg build-essential gcc g++ \
#     htop unzip openssh-server screen && \
#     git lfs install --force && \
#     apt-get clean && \
#     apt-get -y autoremove && \
#     rm -rf /var/lib/apt/lists/*

# # 安装 code-server 和 vscode 常用插件（合并）
# RUN curl -fsSL https://code-server.dev/install.sh | sh && \
#     code-server --install-extension redhat.vscode-yaml && \
#     code-server --install-extension dbaeumer.vscode-eslint && \
#     code-server --install-extension eamodio.gitlens && \
#     code-server --install-extension tencent-cloud.coding-copilot  && \
#     code-server --install-extension ms-python.python

# # 指定字符集
# ENV LANG C.UTF-8
# ENV LANGUAGE C.UTF-8

# # 安装 pip 包（合并所有pip安装，使用清华源，移除重复）
# RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
#     pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     modelscope "huggingface_hub[cli]" jupyterlab torch coscmd \
#     datasets accelerate evaluate scikit-learn rouge_score lm_eval ftfy matplotlib 

FROM docker.cnb.cool/aisshina/sparse-load
COPY docker_context/arkvale /opt/conda/envs/arkvale
# 3. 设置环境变量，让系统知道这个新环境的存在
ENV PATH /opt/conda/envs/arkvale/bin:$PATH


