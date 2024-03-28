#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define N 16384  // Matrix size

__global__ void helloFromGPU() {
    printf("Hello from GPU!\n");
}

__global__ void matrixMultiply(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i]*B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    float *d_A, *d_B, *d_C;  // Device copies of A, B, C
    int size = N * N * sizeof(float);

    // Cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Initialize host copies of A, B
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < 1; i++) {
        for (int i = 0; i < N*N; i++) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        auto start = std::chrono::high_resolution_clock::now();
        // Copy inputs to device
        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
        // cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        printf("To cuda: %ld micros\n", duration.count());


        // Enable TF32 tensor core math mode
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    
        float alpha = 1.0f;
        float beta = 0.0f;

        start = std::chrono::high_resolution_clock::now();
        // Launch kernel on the GPU
        //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, CUDA_R_32F, N, d_B, CUDA_R_32F, N, &beta, d_C, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DFALT);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        printf("Multiply cublas: %ld micros\n", duration.count());
        

        start = std::chrono::high_resolution_clock::now();
        // Launch kernel on the GPU
        dim3 threadsPerBlock(32, 32, 1);
        dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
        matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        printf("Multiply mine: %ld micros\n", duration.count());

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        } else {
            std::cout << "Kernel launch successful!" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        // Copy result back to host
        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        printf("To cpu: %ld micros\n", duration.count());
    }

    // //Print result
    // for (int i = 0; i < N * N; i++) {
    //     printf("%f   ", C[i]);
    //     if ((i + 1) % N == 0) printf("\n");
    // }

    // Cleanup
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}

