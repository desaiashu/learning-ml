#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>
#include <functional>
#include <arm_neon.h>
// #define NS_PRIVATE_IMPLEMENTATION
// #define CA_PRIVATE_IMPLEMENTATION
// #define MTL_PRIVATE_IMPLEMENTATION
// #include <Foundation/Foundation.hpp>
// #include <Metal/Metal.hpp>
// #include <QuartzCore/QuartzCore.hpp>

// Function to calculate the determinant of a matrix
double det(double** m, int size) {
    double det = 1;
    for (int row = 0; row < size; ++row) {
        int swap = row + 1;
        while (m[row][row] == 0) {
            if (swap == size) {
                return 0;
            }
            std::swap(m[row], m[swap]);
            det *= -1;
            swap++;
        }
        for (int op = row + 1; op < size; ++op) {
            for (int col = row + 1; col < size; ++col) {
                m[op][col] -= m[row][col] * m[op][row] / m[row][row];
            }
            m[op][row] = 0;
        }
    }
    for (int i = 0; i < size; ++i) {
        det *= m[i][i];
    }
    return det;
}

void multiply_row(double** m1, double** m2, double** result, int row, int cols1, int cols2) {
    for (int c = 0; c < cols2; ++c) {
        result[row][c] = 0; // Initialize the element
        for (int d = 0; d < cols1; ++d) {
            result[row][c] += m1[row][d] * m2[d][c];
        }
    }
}

void multiply_row_simd(double** m1, double** m2, double** result, int row, int cols1, int cols2) {

    int block = 512;

    for (int c = 0; c < cols2; ++c) {
        float64x2_t sum = vdupq_n_f64(0); // Initialize the sum to zero
        int d;
        for (d = 0; d <= cols1 - block; d += block) {
            float64x2_t vec1 = vld1q_f64(&m1[row][d]);
            float64x2_t vec2 = vld1q_f64(&m2[d][c]);
            sum = vaddq_f64(sum, vmulq_f64(vec1, vec2));
        }
        // Handle the tail case if cols1 is not a multiple of block size
        for (; d < cols1; ++d) {
            sum = vaddq_f64(sum, vmulq_n_f64(vld1q_dup_f64(&m1[row][d]), m2[d][c]));
        }
        result[row][c] = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    }
}

double** mult(double** m1, int rows1, int cols1, double** m2, int rows2, int cols2) {
    if (cols1 != rows2) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    // Allocate memory for the result matrix
    double** result = new double*[rows1];
    for (int i = 0; i < rows1; ++i) {
        result[i] = new double[cols2]();
    }

    // Get the number of supported concurrent threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "Threads: " << num_threads << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Function to process a chunk of rows
    auto process_chunk = [&](int start_row, int end_row) {
        for (int r = start_row; r < end_row; ++r) {
            multiply_row_simd(m1, m2, result, r, cols1, cols2);
        }
    };

    // Launch threads to process chunks of rows
    int chunk_size = rows1 / num_threads;
    int start_row = 0;
    for (unsigned int i = 0; i < num_threads - 1; ++i) {
        int end_row = start_row + chunk_size;
        threads.emplace_back(process_chunk, start_row, end_row);
        start_row = end_row;
    }
    // Last thread may take extra if rows1 is not divisible by num_threads
    threads.emplace_back(process_chunk, start_row, rows1);

    // Wait for all threads to complete
    for (std::thread& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    return result;
}

// Function to generate a random matrix
double** rand_matrix(int rows, int cols) {
    double** matrix = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[cols];
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

// Function to delete a matrix and free memory
void delete_matrix(double** matrix, int size) {
    for (int i = 0; i < size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr))); // Seed for random number generation

    int size = 2048; // Example matrix size
    double** matrix = rand_matrix(size, size);

    auto start = std::chrono::high_resolution_clock::now();

    double matrix_det;

    for (int i = 0; i < 1; ++i) {
        matrix_det = det(matrix, size);
    }
    // double matrix_det = det(matrix, size);

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Determinant: " << matrix_det << std::endl;
    std::cout << "Time taken by determinant: " << duration.count() << " microseconds" << std::endl;

    delete_matrix(matrix, size); // Clean up memory


    double** m1 = rand_matrix(size, size);
    double** m2 = rand_matrix(size, size);

    start = std::chrono::high_resolution_clock::now();

    mult(m1, size, size, m2, size, size);

    // Stop timing
    stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by multiplication: " << duration.count() << " microseconds" << std::endl;

    delete_matrix(m1, size); // Clean up memory
    delete_matrix(m2, size); // Clean up memory

    return 0;
}