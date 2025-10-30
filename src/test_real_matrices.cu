#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include "../include/formats/matrix_formats.hpp"
#include "../include/formats/cuda_matrix_formats.cuh"
#include "../include/io/matrix_market.hpp"

// Forward declarations
CSRMatrix lanczos_cpu_reference(const CSRMatrix& input, double scale_x, double scale_y, double threshold);

void test_with_matrix(const std::string& filename, double scale, double threshold) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << filename << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    try {
        // Check if file exists
        if (!std::filesystem::exists(filename)) {
            std::cout << "File not found: " << filename << std::endl;
            std::cout << "Download matrices first or use synthetic data" << std::endl;
            return;
        }

        // Load matrix from file
        CSRMatrix input_csr = MatrixMarketReader::load_csr(filename);
        std::cout << "Loaded: " << input_csr.rows << " x " << input_csr.cols
                  << ", NNZ: " << input_csr.nnz
                  << " (density: " << (100.0 * input_csr.nnz / (input_csr.rows * input_csr.cols)) << "%)" << std::endl;

        // CPU test
        std::cout << "\n--- CPU Reference ---" << std::endl;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        CSRMatrix cpu_output = lanczos_cpu_reference(input_csr, scale, scale, threshold);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

        std::cout << "Time: " << cpu_time.count() << " ms" << std::endl;
        std::cout << "Output: " << cpu_output.rows << " x " << cpu_output.cols
                  << ", NNZ: " << cpu_output.nnz
                  << " (density: " << (100.0 * cpu_output.nnz / (cpu_output.rows * cpu_output.cols)) << "%)" << std::endl;

        // GPU test
        std::cout << "\n--- GPU Implementation ---" << std::endl;
        CSRDeviceMatrix device_input;

        auto gpu_start = std::chrono::high_resolution_clock::now();
        allocate_device_matrix(input_csr, device_input);

        CSRMatrix gpu_output;
        lanczos_gpu(device_input, gpu_output, scale, scale, threshold);

        free_device_matrix(device_input);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

        std::cout << "Time: " << gpu_time.count() << " ms" << std::endl;
        std::cout << "Output: " << gpu_output.rows << " x " << gpu_output.cols
                  << ", NNZ: " << gpu_output.nnz
                  << " (density: " << (100.0 * gpu_output.nnz / (gpu_output.rows * gpu_output.cols)) << "%)" << std::endl;

        // Results
        std::cout << "\n--- Results Summary ---" << std::endl;
        bool dimensions_match = (cpu_output.rows == gpu_output.rows && cpu_output.cols == gpu_output.cols);
        bool nnz_similar = (std::abs(cpu_output.nnz - gpu_output.nnz) < std::max(cpu_output.nnz, gpu_output.nnz) * 0.1);

        std::cout << "Dimensions match: " << (dimensions_match ? "Yes" : "No") << std::endl;
        std::cout << "NNZ similar: " << (nnz_similar ? "Yes" : "No") << std::endl;

        if (dimensions_match && gpu_time.count() > 0 && cpu_time.count() > 0) {
            double speedup = static_cast<double>(cpu_time.count()) / gpu_time.count();
            std::cout << "Speedup: " << speedup << "x" << std::endl;

            if (speedup > 1.0) {
                std::cout << "GPU is " << speedup << "x faster than CPU!" << std::endl;
            } else {
                std::cout << "CPU is " << (1.0/speedup) << "x faster than GPU" << std::endl;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error processing " << filename << ": " << e.what() << std::endl;
    }
}

void test_synthetic_matrices() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing with Synthetic Matrices" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Create some synthetic test matrices
    std::vector<std::pair<std::string, arma::sp_mat>> synthetic_matrices;

    // Small diagonal matrix
    arma::sp_mat small_diag(30, 30);
    for (int i = 0; i < 30; i++) {
        small_diag(i, i) = 1.0 + i * 0.1;
    }
    synthetic_matrices.push_back({"Small Diagonal (30x30)", small_diag});

    // Medium pattern matrix
    arma::sp_mat medium_pattern(60, 60);
    for (int i = 0; i < 60; i += 2) {
        for (int j = 0; j < 60; j += 3) {
            if ((i + j) % 5 != 0) {
                medium_pattern(i, j) = 1.0 + (i + j) * 0.01;
            }
        }
    }
    synthetic_matrices.push_back({"Medium Pattern (60x60)", medium_pattern});

    // Test each synthetic matrix
    for (const auto& [name, matrix] : synthetic_matrices) {
        CSRMatrix input_csr = CSRMatrix::from_arma(matrix);

        std::cout << "\n" << name << ": " << input_csr.rows << "x" << input_csr.cols
                  << " (NNZ: " << input_csr.nnz << ")" << std::endl;

        // Test with scale 2.0
        double scale = 2.0;
        double threshold = 1e-6;

        // CPU
        auto cpu_start = std::chrono::high_resolution_clock::now();
        CSRMatrix cpu_output = lanczos_cpu_reference(input_csr, scale, scale, threshold);
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - cpu_start);

        // GPU
        auto gpu_start = std::chrono::high_resolution_clock::now();
        CSRDeviceMatrix device_input;
        allocate_device_matrix(input_csr, device_input);
        CSRMatrix gpu_output;
        lanczos_gpu(device_input, gpu_output, scale, scale, threshold);
        free_device_matrix(device_input);
        auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - gpu_start);

        std::cout << "CPU: " << cpu_time.count() << "ms, GPU: " << gpu_time.count() << "ms";
        if (gpu_time.count() > 0) {
            std::cout << " -> " << (double)cpu_time.count()/gpu_time.count() << "x speedup" << std::endl;
        } else {
            std::cout << std::endl;
        }
    }
}

int main() {
    std::cout << "GPU-MatGen: Real Matrix Testing" << std::endl;
    std::cout << "===============================" << std::endl;

    // Create results directory
    std::filesystem::create_directories("results");

    // Test parameters
    double scale = 1.5;
    double threshold = 1e-6;

    // Test with real matrices
    std::vector<std::string> test_matrices = {
        RESOURCES_PATH"1138_bus.mtx",
        RESOURCES_PATH"bcspwr01.mtx",
        RESOURCES_PATH"bayer02.mtx",
        RESOURCES_PATH"FEM_3D_thermal1.mtx",
        RESOURCES_PATH"lpl3.mtx",
        RESOURCES_PATH"poli2.mtx"
    };

    bool any_real_matrices_found = false;

    for (const auto& matrix_file : test_matrices) {
        if (std::filesystem::exists(matrix_file)) {
            any_real_matrices_found = true;
            test_with_matrix(matrix_file, scale, threshold);
        }
    }

    // If no real matrices found, test with synthetic ones
    if (!any_real_matrices_found) {
        std::cout << "\n💡 No real matrix files found. Testing with synthetic matrices..." << std::endl;
        test_synthetic_matrices();
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing complete!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}