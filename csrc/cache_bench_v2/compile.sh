export Torch_DIR=/NVME4/conda/envs/router_qys/lib/python3.9/site-packages/torch/share/cmake/Torch
export LD_LIBRARY_PATH=/NVME4/conda/envs/router_qys/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH


# x299a
export Torch_DIR=/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# 进入 build
rm -rf build
mkdir build
cd build
cmake -DTorch_DIR=$Torch_DIR ..
cmake --build . --config Release 