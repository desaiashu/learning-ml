#include <arm_neon.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <time.h>

#define TILE_SIZE 32 // Define an appropriate tile size for your system's cache

void matrix_multiply(float *A, float *B, float *C, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                int max_i = i + TILE_SIZE < M ? i + TILE_SIZE : M;
                int max_j = j + TILE_SIZE < N ? j + TILE_SIZE : N;
                int max_k = k + TILE_SIZE < K ? k + TILE_SIZE : K;
                for (int ii = i; ii < max_i; ++ii) {
                    for (int jj = j; jj < max_j; ++jj) {
                        float32x4_t c = vdupq_n_f32(0);
                        for (int kk = k; kk < max_k; kk += 4) {
                            float32x4_t a = vld1q_f32(A + ii * K + kk);
                            float32x4_t b = vld1q_f32(B + kk * N + jj);
                            c = vfmaq_f32(c, a, b);
                        }
                        C[ii * N + jj] += vaddvq_f32(c);
                    }
                }
            }
        }
    }
}

void mult(float *A, float *B, float *C, int M, int N, int K) {

    // for (int i = 0; i < M*N*K; i++) {

    //     C[i] = 0;
    // }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void initialize_matrix_with_random_doubles(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    clock_t start, end;
    
    // Example matrices dimensions
    const int M = 1024; // A rows
    const int N = 1024; // B columns
    const int K = 1024; // A columns and B rows

    // Allocate memory for matrices
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));

    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize matrices A and B with random doubles
    initialize_matrix_with_random_doubles(A, M, K);
    initialize_matrix_with_random_doubles(B, K, N);

    // Start timing
    start = clock();

    // Perform matrix multiplication
    mult(A, B, C, M, N, K);

    // Stop timing
    end = clock();

    // Calculate the elapsed time in microseconds
    long micros = ((double) (end - start)) / CLOCKS_PER_SEC * 1000000;

    printf("The first entry is %f\n", C[0]);
    printf("The operation took %ld microseconds\n", micros);

    // Clean up
    free(A);
    free(B);
    free(C);

    return 0;
}