cmake -DTorch_DIR=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release