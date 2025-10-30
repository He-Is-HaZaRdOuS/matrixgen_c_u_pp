#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "../../include/kernels/lanczos.cuh"
#include "../../include/formats/cuda_matrix_formats.cuh"

__global__ void lanczos_kernel_2d(
    const int* row_ptr, const int* col_ind, const double* values,
    int input_rows, int input_cols,
    double* temp_dense, int output_rows, int output_cols,
    double scale_x, double scale_y, double threshold) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_rows || j >= output_cols) return;

    double orig_i = i / scale_y;
    double orig_j = j / scale_x;

    double sum = 0.0;
    double weight_sum = 0.0;

    // Lanczos window
    int start_i = max(0, (int)floor(orig_i - 2));
    int end_i = min(input_rows - 1, (int)ceil(orig_i + 2));
    int start_j = max(0, (int)floor(orig_j - 2));
    int end_j = min(input_cols - 1, (int)ceil(orig_j + 2));

    for (int ii = start_i; ii <= end_i; ii++) {
        // Get row information from CSR
        int row_start = row_ptr[ii];
        int row_end = row_ptr[ii + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int jj = col_ind[idx];

            // Check if this column is in our window
            if (jj >= start_j && jj <= end_j) {
                double val = values[idx];
                double dist_i = orig_i - ii;
                double dist_j = orig_j - jj;

                double weight = lanczos_kernel(dist_i) * lanczos_kernel(dist_j);

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
    double scale_x, double scale_y, double threshold) {

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
        scale_x, scale_y, threshold
    );

    cudaDeviceSynchronize();
}

// Wrapper function
void lanczos_gpu(
    const CSRDeviceMatrix& device_input,
    CSRMatrix& output,
    double scale_x, double scale_y, double threshold) {

    int output_rows = round(device_input.rows * scale_y);
    int output_cols = round(device_input.cols * scale_x);

    // Allocate temporary dense output on device
    double* d_temp_dense;
    size_t dense_size = output_rows * output_cols * sizeof(double);
    cudaMalloc(&d_temp_dense, dense_size);
    cudaMemset(d_temp_dense, 0, dense_size);

    // Launch the kernel
    lanczos_gpu_launcher(
        device_input.row_ptr, device_input.col_ind, device_input.values,
        device_input.rows, device_input.cols,
        d_temp_dense, output_rows, output_cols,
        scale_x, scale_y, threshold
    );

    // Copy results back to host
    std::vector<double> temp_host(output_rows * output_cols);
    cudaMemcpy(temp_host.data(), d_temp_dense, dense_size, cudaMemcpyDeviceToHost);

    cudaFree(d_temp_dense);

    // Simple CPU compaction
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

    output.rows = output_rows;
    output.cols = output_cols;
    output.nnz = vals.size();
    output.values = vals;
    output.col_ind = cols;

    // Build row_ptr
    output.row_ptr.resize(output_rows + 1, 0);
    for (size_t i = 0; i < rows.size(); i++) {
        output.row_ptr[rows[i] + 1]++;
    }
    for (int i = 0; i < output_rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }
}