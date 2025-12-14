#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <chrono>

#define BATCH_SIZE 1
#define NUM_HEADS 64
#define QUERY_LEN 128
#define DIM 128
#define MAX_KEY_LEN 4096

// CUDA Error Check
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Random initialization of matrices
void initialize_random(float* d_ptr, size_t size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_ptr, size);
    curandDestroyGenerator(gen);
}

// Matrix multiplication using cuBLAS for SDPA
void sdpa_cublas(float* Q, float* K, float* V, float* output, int batch_size, int num_heads, int query_len, int key_len, int dim) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f / sqrtf(dim);
    float beta = 0.0f;
    
    float* d_scores;
    float* d_attn;
    CHECK_CUDA(cudaMalloc((void**)&d_scores, batch_size * num_heads * query_len * key_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_attn, batch_size * num_heads * query_len * key_len * sizeof(float)));
    
    // Q * K^T
    cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, key_len, query_len, dim, &alpha, K, dim, key_len * dim, Q, dim, query_len * dim, &beta, d_scores, key_len, query_len * key_len, batch_size * num_heads);
    
    // Softmax (simple scaling for now)
    CHECK_CUDA(cudaMemcpy(d_attn, d_scores, batch_size * num_heads * query_len * key_len * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Attention * V
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, query_len, key_len, &alpha, V, dim, key_len * dim, d_attn, key_len, query_len * key_len, &beta, output, dim, query_len * dim, batch_size * num_heads);
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_attn));
    cublasDestroy(handle);
}

int main() {
    // Allocate memory
    float *d_Q, *d_K, *d_V, *d_output;
    size_t Q_size = BATCH_SIZE * NUM_HEADS * QUERY_LEN * DIM;
    size_t K_size = BATCH_SIZE * NUM_HEADS * MAX_KEY_LEN * DIM;
    size_t V_size = BATCH_SIZE * NUM_HEADS * MAX_KEY_LEN * DIM;
    size_t output_size = BATCH_SIZE * NUM_HEADS * QUERY_LEN * DIM;
    
    CHECK_CUDA(cudaMalloc((void**)&d_Q, Q_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_K, K_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_V, V_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, output_size * sizeof(float)));
    
    // Initialize data
    initialize_random(d_Q, Q_size);
    initialize_random(d_K, K_size);
    initialize_random(d_V, V_size);
    
    // Warm-up runs
    for (int i = 0; i < 5; ++i) {
        sdpa_cublas(d_Q, d_K, d_V, d_output, BATCH_SIZE, NUM_HEADS, QUERY_LEN, MAX_KEY_LEN, DIM);
    }
    cudaDeviceSynchronize();
    
    // Benchmark SDPA
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {  // Run multiple times for averaging
        sdpa_cublas(d_Q, d_K, d_V, d_output, BATCH_SIZE, NUM_HEADS, QUERY_LEN, MAX_KEY_LEN, DIM);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "SDPA cuBLAS Execution Time: " << std::chrono::duration<float, std::milli>(end - start).count() / 10 << " ms (average)" << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_output));
    
    return 0;
}