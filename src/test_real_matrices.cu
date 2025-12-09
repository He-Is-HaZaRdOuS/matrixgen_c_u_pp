#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#include "formats/matrix_formats.hpp"
#include "formats/cuda_matrix_formats.cuh"
#include "io/matrix_market.hpp"
#include "io/matrix_visualizer.hpp"
#include "utils/cuda_matrix_utils.cuh"

/* Prototypes */
CSRMatrix lanczos_cpu_reference(const CSRMatrix& input, double scale_x, double scale_y, int a, double threshold);
CSRMatrix lanczos_sparse_deterministic(const CSRMatrix& input, double scale_x, double scale_y, int a, double threshold);
CSRMatrix lanczos_sparse_omp(const CSRMatrix& input, double scale_x, double scale_y, int a, double threshold);

void benchmark_matrix(const std::string& filename, double scale, int a, double threshold) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing: " << filename << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    /* Load matrix from file */
    CSRMatrix input_csr = MatrixMarketReader::load_csr(filename);
    std::cout << "Loaded: " << input_csr.rows << " x " << input_csr.cols
              << ", NNZ: " << input_csr.nnz
              << " (density: " << (100.0 * input_csr.nnz / (input_csr.rows * input_csr.cols)) << "%)" << std::endl;

    // Visualize input
    std::filesystem::create_directories(RESOURCES_PATH"plots");
    std::string base_name = std::filesystem::path(filename).stem().string();
    // MatrixVisualizer::save_spy_plot(input_csr, RESOURCES_PATH"plots/" + base_name + "_input");
    // MatrixVisualizer::save_heatmap(input_csr, RESOURCES_PATH"plots/" + base_name + "_input_heatmap");

    // // CPU test
    // std::cout << "\n--- CPU Reference ---" << std::endl;
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // CSRMatrix cpu_output = lanczos_cpu_reference(input_csr, scale, scale, a, threshold);
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    //
    // std::cout << "Time: " << cpu_time.count() << " ms" << std::endl;
    // std::cout << "Output: " << cpu_output.rows << " x " << cpu_output.cols
    //           << ", NNZ: " << cpu_output.nnz
    //           << " (density: " << (100.0 * cpu_output.nnz / (cpu_output.rows * cpu_output.cols)) << "%)" << std::endl;
    //
    // // Visualize CPU output
    // MatrixVisualizer::save_spy_plot(cpu_output, RESOURCES_PATH"plots/" + base_name + "_cpu_output");
    // MatrixVisualizer::save_heatmap(cpu_output, RESOURCES_PATH"plots/" + base_name + "_cpu_output_heatmap");

    //CPU sparse test
    std::cout << "\n--- CPU Sparse ---" << std::endl;
    auto cpu_sparse_start = std::chrono::high_resolution_clock::now();
    CSRMatrix cpu_sparse_output = lanczos_sparse_omp(input_csr, scale, scale, a, threshold);
    auto cpu_sparse_end = std::chrono::high_resolution_clock::now();
    auto cpu_sparse_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_sparse_end - cpu_sparse_start);

    std::cout << "Time: " << cpu_sparse_time.count() << " ms" << std::endl;
    std::cout << "Output: " << cpu_sparse_output.rows << " x " << cpu_sparse_output.cols
              << ", NNZ: " << cpu_sparse_output.nnz
              << " (density: " << (100.0 * cpu_sparse_output.nnz / (cpu_sparse_output.rows * cpu_sparse_output.cols)) << "%)" << std::endl;

    // Visualize CPU sparse output
    // MatrixVisualizer::save_spy_plot(cpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_cpu_sparse_output");
    // MatrixVisualizer::save_heatmap(cpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_cpu_sparse_output_heatmap");

    // // GPU test
    // std::cout << "\n--- GPU Implementation ---" << std::endl;
    // CSRDeviceMatrix device_input;
    //
    // auto gpu_start = std::chrono::high_resolution_clock::now();
    // allocate_device_matrix(input_csr, device_input);
    //
    // CSRMatrix gpu_output;
    // lanczos_gpu(device_input, gpu_output, scale, scale, a, threshold);
    //
    // free_device_matrix(device_input);
    // auto gpu_end = std::chrono::high_resolution_clock::now();
    // auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
    //
    // std::cout << "Time: " << gpu_time.count() << " ms" << std::endl;
    // std::cout << "Output: " << gpu_output.rows << " x " << gpu_output.cols
    //           << ", NNZ: " << gpu_output.nnz
    //           << " (density: " << (100.0 * gpu_output.nnz / (gpu_output.rows * gpu_output.cols)) << "%)" << std::endl;
    //
    // // Visualize GPU output
    // MatrixVisualizer::save_spy_plot(gpu_output, RESOURCES_PATH"plots/" + base_name + "_gpu_output");
    // MatrixVisualizer::save_heatmap(gpu_output, RESOURCES_PATH"plots/" + base_name + "_gpu_output_heatmap");

    // GPU sparse test
    // std::cout << "\n--- GPU sparse Implementation ---" << std::endl;
    // CSRDeviceMatrix device_input_sparse;
    //
    // auto gpu_sparse_start = std::chrono::high_resolution_clock::now();
    // allocate_device_matrix(input_csr, device_input_sparse);
    //
    // CSRMatrix gpu_sparse_output;
    // lanczos_sparse_gpu(device_input_sparse, gpu_sparse_output, scale, scale, a, threshold);
    //
    // free_device_matrix(device_input_sparse);
    // auto gpu_sparse_end = std::chrono::high_resolution_clock::now();
    // auto gpu_sparse_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_sparse_end - gpu_sparse_start);
    //
    // std::cout << "Time: " << gpu_sparse_time.count() << " ms" << std::endl;
    // std::cout << "Output: " << gpu_sparse_output.rows << " x " << gpu_sparse_output.cols
    //           << ", NNZ: " << gpu_sparse_output.nnz
    //           << " (density: " << (100.0 * gpu_sparse_output.nnz / (gpu_sparse_output.rows * gpu_sparse_output.cols)) << "%)" << std::endl;

    // Visualize GPU output
    // MatrixVisualizer::save_spy_plot(gpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_gpu_sparse_output");
    // MatrixVisualizer::save_heatmap(gpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_gpu_sparse_output_heatmap");

    // GPU sparse test
    std::cout << "\n--- GPU sparse !2! Implementation ---" << std::endl;
    CSRDeviceMatrix device_input_sparse;

    auto gpu_sparse_start = std::chrono::high_resolution_clock::now();
    allocate_device_matrix(input_csr, device_input_sparse);

    CSRMatrix gpu_sparse_output = CSRMatrix();
    lanczos_sparse_gpu_improved_chunked(device_input_sparse, gpu_sparse_output, scale, scale, a, threshold);

    free_device_matrix(device_input_sparse);
    auto gpu_sparse_end = std::chrono::high_resolution_clock::now();
    auto gpu_sparse_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_sparse_end - gpu_sparse_start);

    std::cout << "Time: " << gpu_sparse_time.count() << " ms" << std::endl;
    std::cout << "Output: " << gpu_sparse_output.rows << " x " << gpu_sparse_output.cols
              << ", NNZ: " << gpu_sparse_output.nnz
              << " (density: " << (100.0 * gpu_sparse_output.nnz / (gpu_sparse_output.rows * gpu_sparse_output.cols)) << "%)" << std::endl;

    // Save GPU sparse output to disk
    // std::string out_dir = "results";
    // std::filesystem::create_directories(out_dir);
    //
    // std::string out_name = base_name
    //                      + "_scale" + std::to_string(scale)
    //                      + "_a" + std::to_string(a)
    //                      + "_gpu_sparse2.mtx";
    //
    // std::string out_path = out_dir + "/" + out_name;
    // MatrixMarketReader::save_csr(gpu_sparse_output, out_path);
    // std::cout << "Saved GPU sparse output to: " << out_path << std::endl;


    // Visualize GPU output
    // MatrixVisualizer::save_spy_plot(gpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_gpu_sparse2_output");
    // MatrixVisualizer::save_heatmap(gpu_sparse_output, RESOURCES_PATH"plots/" + base_name + "_gpu_sparse2_output_heatmap");

    // // Results
    // std::cout << "\n--- Results Summary ---" << std::endl;
    // bool dimensions_match = (cpu_output.rows == gpu_output.rows && cpu_output.cols == gpu_output.cols);
    // bool nnz_similar = (std::abs(cpu_output.nnz - gpu_output.nnz) < std::max(cpu_output.nnz, gpu_output.nnz) * 0.1);
    //
    // std::cout << "Dimensions match: " << (dimensions_match ? "Yes" : "No") << std::endl;
    // std::cout << "NNZ similar: " << (nnz_similar ? "Yes" : "No") << std::endl;
    //
    // if (dimensions_match && gpu_time.count() > 0 && cpu_time.count() > 0) {
    //     double speedup = static_cast<double>(cpu_time.count()) / gpu_time.count();
    //     std::cout << "Speedup: " << speedup << "x" << std::endl;
    //
    //     if (speedup > 1.0) {
    //         std::cout << "GPU is " << speedup << "x faster than CPU!" << std::endl;
    //     } else {
    //         std::cout << "CPU is " << (1.0/speedup) << "x faster than GPU" << std::endl;
    //     }
    // }

    // Results
    // std::cout << "\n--- Results Summary ---" << std::endl;
    // bool dimensions_match = (cpu_sparse_output.rows == gpu_sparse_output.rows && cpu_sparse_output.cols == gpu_sparse_output.cols);
    // bool nnz_similar = (std::abs(cpu_sparse_output.nnz - gpu_sparse_output.nnz) < std::max(cpu_sparse_output.nnz, gpu_sparse_output.nnz) * 0.1);
    //
    // std::cout << "Dimensions match: " << (dimensions_match ? "Yes" : "No") << std::endl;
    // std::cout << "NNZ similar: " << (nnz_similar ? "Yes" : "No") << std::endl;
    //
    // if (dimensions_match && gpu_sparse_time.count() > 0 && cpu_sparse_time.count() > 0) {
    //     double speedup = static_cast<double>(cpu_sparse_time.count()) / gpu_sparse_time.count();
    //     std::cout << "Speedup: " << speedup << "x" << std::endl;
    //
    //     if (speedup > 1.0) {
    //         std::cout << "GPU is " << speedup << "x faster than CPU!" << std::endl;
    //     } else {
    //         std::cout << "CPU is " << (1.0/speedup) << "x faster than GPU" << std::endl;
    //     }
    // }
}

int main() {
    std::cout << "GPU-MatGen: Real Matrix Testing" << std::endl;
    std::cout << "===============================" << std::endl;

    /* Create results directory */
    std::filesystem::create_directories("results");

    /* Test parameters */
    std::vector<double> scales = {0.5, 2, 4};
    int a = 2;
    double threshold = 1e-6;

    /* real matrix files on disk */
    std::vector<std::string> test_matrices = {
        // RESOURCES_PATH"1138_bus.mtx",
        // RESOURCES_PATH"bayer02.mtx",
        // RESOURCES_PATH"FEM_3D_thermal1.mtx",
        // RESOURCES_PATH"lpl3.mtx",
        // RESOURCES_PATH"poli3.mtx",
        // RESOURCES_PATH"LFAT5.mtx",
        // RESOURCES_PATH"para-7.mtx",
        RESOURCES_PATH"bcsstk30.mtx",
    };

    for (const auto &s: scales) {
        for (const auto& matrix_file : test_matrices) {
            if (std::filesystem::exists(matrix_file)) {
                benchmark_matrix(matrix_file, s, a, threshold);
            }
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing complete!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
}