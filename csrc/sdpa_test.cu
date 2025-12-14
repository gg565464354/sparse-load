#include <iostream>
#include <vector>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>

#define BATCH_SIZE 1
#define NUM_HEADS 64
#define QUERY_LEN 128
#define DIM 128
#define MAX_KEY_LEN 4096

// **SDPA 计算**
torch::Tensor sdpa_libtorch(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    return at::scaled_dot_product_attention(Q, K, V);
}

int main() {
    torch::manual_seed(1234);
    torch::Device device(torch::kCUDA);

    // **初始化 Q, K, V**
    auto Q = torch::randn({BATCH_SIZE, NUM_HEADS, QUERY_LEN, DIM}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    auto K = torch::randn({BATCH_SIZE, NUM_HEADS, MAX_KEY_LEN, DIM}, torch::TensorOptions().dtype(torch::kFloat16).device(device));
    auto V = torch::randn({BATCH_SIZE, NUM_HEADS, MAX_KEY_LEN, DIM}, torch::TensorOptions().dtype(torch::kFloat16).device(device));

    // **Warm-up**
    for (int i = 0; i < 5; ++i) {
        sdpa_libtorch(Q, K, V);
    }
    torch::cuda::synchronize();

    // **Benchmark**
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        sdpa_libtorch(Q, K, V);
    }
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    float avg_time = std::chrono::duration<float, std::milli>(end - start).count() / 10;
    std::cout << "SDPA LibTorch Execution Time: " << avg_time << " ms (average)" << std::endl;

    return 0;
}
