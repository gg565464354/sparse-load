mkdir build
cd build
cmake -DTorch_DIR=/home/oem/.local/lib/python3.10/site-packages/torch/share/cmake ..
cmake --build . --config Release