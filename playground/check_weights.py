import torch
import os

# 检查权重文件的形状
weights_dir = "/workspace/SparseCache/accuracy/setup/weights/LLaMA-2-7B-32K_0.2"
skew_path = "/workspace/SparseCache/accuracy/setup/skewing_matrix/LLaMA-2-7B-32K.pt"

print("Checking weight file shapes...")

# 检查 skewing_matrix
try:
    A = torch.load(skew_path, map_location="cpu")
    print(f"Skewing matrix shape: {A.shape if hasattr(A, 'shape') else type(A)}")
    if isinstance(A, (list, tuple)):
        print(f"  - Number of layers: {len(A)}")
        if len(A) > 0:
            print(f"  - First layer shape: {A[0].shape}")
    else:
        print(f"  - Total shape: {A.shape}")
except Exception as e:
    print(f"Error loading skewing matrix: {e}")

# 检查前几个 partial_weight_q 文件
for layer_idx in range(3):  # 只检查前3层
    try:
        partial_weight_path = os.path.join(weights_dir, f"partial_weight_q_{layer_idx}.pt")
        if os.path.exists(partial_weight_path):
            weight = torch.load(partial_weight_path, map_location="cpu")
            print(f"Layer {layer_idx} partial_weight_q shape: {weight.shape if hasattr(weight, 'shape') else type(weight)}")
            if hasattr(weight, 'shape') and len(weight.shape) > 0:
                print(f"  - Dimensions: {weight.shape}")
                if isinstance(weight, (list, tuple)):
                    print(f"  - Number of elements: {len(weight)}")
                    if len(weight) > 0:
                        print(f"  - First element shape: {weight[0].shape if hasattr(weight[0], 'shape') else type(weight[0])}")
        else:
            print(f"Layer {layer_idx}: File not found")
    except Exception as e:
        print(f"Layer {layer_idx} error: {e}")

print("\nDone checking weights.")