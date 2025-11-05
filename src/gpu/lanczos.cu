#include <cuda.h>
#include <vector>
#include <iostream>
#include "formats/cuda_matrix_formats.cuh"
#include "utils/cuda_utils.cuh"
#include "kernels/lanczos.hpp"

__global__ void lanczos_kernel_2d(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    int input_rows, int input_cols,
    double* __restrict__ temp_dense,
    int output_rows, int output_cols,
    double scale_x, double scale_y, int a) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_rows || j >= output_cols) return;

    double orig_i = i / scale_y;
    double orig_j = j / scale_x;

    double sum = 0.0;
    double weight_sum = 0.0;

    int start_i = max(0, (int)floor(orig_i - a));
    int end_i = min(input_rows - 1, (int)ceil(orig_i + a));
    int start_j = max(0, (int)floor(orig_j - a));
    int end_j = min(input_cols - 1, (int)ceil(orig_j + a));

    for (int ii = start_i; ii <= end_i; ii++) {
        int row_start = row_ptr[ii];
        int row_end = row_ptr[ii + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int jj = col_ind[idx];

            if (jj >= start_j && jj <= end_j) {
                double val = values[idx];
                double dist_i = orig_i - ii;
                double dist_j = orig_j - jj;

                // Compute weight directly - let compiler optimize
                double weight = lanczos_kernel(dist_i, a) * lanczos_kernel(dist_j, a);

                // Single branch - less divergence
                if (fabs(weight) > 1e-9) {
                    sum += val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    if (fabs(weight_sum) > 1e-9) {
        temp_dense[i * output_cols + j] = sum / weight_sum;
    } else {
        temp_dense[i * output_cols + j] = 0.0;
    }
}

void lanczos_gpu_launcher(
    const int* row_ptr, const int* col_ind, const double* values,
    int input_rows, int input_cols,
    double* temp_dense, int output_rows, int output_cols,
    double scale_x, double scale_y, int a) {

    // Configure kernel launch
    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_cols + block_size.x - 1) / block_size.x,
        (output_rows + block_size.y - 1) / block_size.y
    );

    // Launch kernel
    lanczos_kernel_2d<<<grid_size, block_size>>>(
        row_ptr, col_ind, values,
        input_rows, input_cols,
        temp_dense, output_rows, output_cols,
        scale_x, scale_y, a
    );

    cudaDeviceSynchronize();
}

void lanczos_gpu(const CSRDeviceMatrix& device_input,
                CSRMatrix& output,
                double scale_x, double scale_y, int a, double threshold) {

    int output_rows = round(device_input.rows * scale_y);
    int output_cols = round(device_input.cols * scale_x);

    // Allocation
    auto alloc_start = std::chrono::high_resolution_clock::now();
    double* d_temp_dense;
    size_t dense_size = output_rows * output_cols * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_temp_dense, dense_size));
    CUDA_CHECK(cudaMemset(d_temp_dense, 0, dense_size));
    auto alloc_end = std::chrono::high_resolution_clock::now();

    dim3 block_size(16, 16);
    dim3 grid_size(
        (output_cols + block_size.x - 1) / block_size.x,
        (output_rows + block_size.y - 1) / block_size.y
    );

    // Kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lanczos_kernel_2d<<<grid_size, block_size>>>(
        device_input.row_ptr, device_input.col_ind, device_input.values,
        device_input.rows, device_input.cols,
        d_temp_dense, output_rows, output_cols,
        scale_x, scale_y, a
    );
    cudaEventRecord(stop);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start, stop);
    std::cout << "  Kernel time: " << kernel_ms << " ms" << std::endl;

    // Memory copy
    auto copy_start = std::chrono::high_resolution_clock::now();
    std::vector<double> temp_host(output_rows * output_cols);
    CUDA_CHECK(cudaMemcpy(temp_host.data(), d_temp_dense, dense_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_temp_dense));
    auto copy_end = std::chrono::high_resolution_clock::now();

    // Compaction
    auto compact_start = std::chrono::high_resolution_clock::now();
    std::vector<int> rows, cols;
    std::vector<double> vals;

    for (int i = 0; i < output_rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            double val = temp_host[i * output_cols + j];
            if (fabs(val) > threshold) {
                rows.push_back(i);
                cols.push_back(j);
                vals.push_back(val);
            }
        }
    }
    auto compact_end = std::chrono::high_resolution_clock::now();

    auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(alloc_end - alloc_start);
    auto copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(copy_end - copy_start);
    auto compact_time = std::chrono::duration_cast<std::chrono::milliseconds>(compact_end - compact_start);

    std::cout << "  Allocation time: " << alloc_time.count() << " ms" << std::endl;
    std::cout << "  Copy time: " << copy_time.count() << " ms" << std::endl;
    std::cout << "  Compaction time: " << compact_time.count() << " ms" << std::endl;

    // Build output CSR...
    output.rows = output_rows;
    output.cols = output_cols;
    output.nnz = vals.size();
    output.values = vals;
    output.col_ind = cols;

    output.row_ptr.resize(output_rows + 1, 0);
    for (size_t i = 0; i < rows.size(); i++) {
        output.row_ptr[rows[i] + 1]++;
    }
    for (int i = 0; i < output_rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}