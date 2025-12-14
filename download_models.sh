# cd /share/models
# GIT_LFS_SKIP_SMUDGE=1 git clone https://cnb.cool/ai-models/facebook/opt-6.7b
# cd opt-6.7b 
# git lfs pull --include "pytorch_model*"
mkdir -p /share/models
cd /share/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://cnb.cool/ai-models/gradientai/Llama-3-8B-Instruct-262k
cd Llama-3-8B-Instruct-262k
git lfs pull
# mkdir -p /share/models
# cd /share/models
# GIT_LFS_SKIP_SMUDGE=1 git clone https://cnb.cool/ai-models/mistralai/Mistral-7B-Instruct-v0.2
# cd Mistral-7B-Instruct-v0.2
# git lfs pull
# git lfs pull --include "pytorch_model*"
# find . -type f -name "model*" -delete