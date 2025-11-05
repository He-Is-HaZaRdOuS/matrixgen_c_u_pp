#include <iostream>
#include <chrono>
#include <armadillo>
#include "formats/matrix_formats.hpp"
#include "formats/cuda_matrix_formats.cuh"
#include "utils/cuda_matrix_utils.cuh"

// Forward declarations
CSRMatrix lanczos_cpu_reference(const CSRMatrix& input, double scale_x, double scale_y, int a, double threshold);

int main() {
    std::cout << "GPU-MatGen: Testing Basic Functionality" << std::endl;

    try {
        // Create a simple test matrix using Armadillo
        arma::sp_mat test_matrix(20, 20);
        test_matrix(1, 1) = 2.5;
        test_matrix(3, 4) = 1.7;
        test_matrix(5, 5) = 3.2;
        test_matrix(10, 10) = 4.1;
        test_matrix(15, 15) = 2.8;

        std::cout << "Test Matrix: " << test_matrix.n_rows << " x " << test_matrix.n_cols
                  << ", NNZ: " << test_matrix.n_nonzero << std::endl;

        // Convert to our CSR format
        CSRMatrix cpu_input = CSRMatrix::from_arma(test_matrix);
        std::cout << "CSR conversion successful: " << cpu_input.rows << " x " << cpu_input.cols
                  << ", NNZ: " << cpu_input.nnz << std::endl;

        // Test CPU implementation
        std::cout << "\nRunning CPU Lanczos reference..." << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();

        CSRMatrix cpu_output = lanczos_cpu_reference(cpu_input, 1.5, 1.5, 3, 1e-6);

        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

        std::cout << "CPU completed in " << cpu_duration.count() << " ms" << std::endl;
        std::cout << "CPU Output: " << cpu_output.rows << " x " << cpu_output.cols
                  << ", NNZ: " << cpu_output.nnz << std::endl;

        // Test GPU implementation
        std::cout << "\nRunning GPU Lanczos..." << std::endl;

        CSRDeviceMatrix device_input;
        allocate_device_matrix(cpu_input, device_input);

        CSRMatrix gpu_output;

        auto gpu_start = std::chrono::high_resolution_clock::now();
        lanczos_gpu(device_input, gpu_output, 1.5, 1.5, 3, 1e-6);
        auto gpu_end = std::chrono::high_resolution_clock::now();

        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

        std::cout << "GPU completed in " << gpu_duration.count() << " ms" << std::endl;
        std::cout << "GPU Output: " << gpu_output.rows << " x " << gpu_output.cols
                  << ", NNZ: " << gpu_output.nnz << std::endl;

        // Calculate speedup if both completed successfully
        if (gpu_duration.count() > 0 && cpu_duration.count() > 0) {
            double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
            std::cout << "\nSpeedup: " << speedup << "x" << std::endl;
        }

        free_device_matrix(device_input);

        std::cout << "\nSUCCESS: Basic functionality working!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}