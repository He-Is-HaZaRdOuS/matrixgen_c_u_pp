#include <algorithm>

#include "kernels/lanczos.hpp"
#include "utils/cuda_utils.cuh"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <vector>
#include <iostream>

#include "formats/cuda_matrix_formats.cuh"

// Binary search to find row index for a given nnz index
__device__ int find_row_for_nnz(const int* row_ptr, int num_rows, int nnz_idx) {
    int left = 0;
    int right = num_rows;

    while (left < right) {
        int mid = (left + right) / 2;
        if (row_ptr[mid] <= nnz_idx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    return left - 1;
}

// Kernel 1: Compute influence zones - each input nonzero affects a window of output pixels
__global__ void compute_influence_zones_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    int input_rows,
    int input_cols,
    int total_nnz,
    double scale_x,
    double scale_y,
    int kernel_size,
    int output_rows,
    int output_cols,
    int* __restrict__ influenced_pixels,     // Output: [row, col] pairs
    int* __restrict__ influence_counts) {    // Output: count per thread

    int nnz_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nnz_idx >= total_nnz) return;

    // Find which (ii, jj) this nnz belongs to via binary search
    int ii = find_row_for_nnz(row_ptr, input_rows, nnz_idx);
    int jj = col_ind[nnz_idx];

    // Calculate output window this input pixel affects
    double out_i_center = ii * scale_y;
    double out_j_center = jj * scale_x;

    // Lanczos support in output space
    int out_i_start = max(0, (int)floor(out_i_center - kernel_size * scale_y));
    int out_i_end = min(output_rows - 1, (int)ceil(out_i_center + kernel_size * scale_y));
    int out_j_start = max(0, (int)floor(out_j_center - kernel_size * scale_x));
    int out_j_end = min(output_cols - 1, (int)ceil(out_j_center + kernel_size * scale_x));

    // Calculate window size
    int window_rows = out_i_end - out_i_start + 1;
    int window_cols = out_j_end - out_j_start + 1;
    int window_size = window_rows * window_cols;

    // Store count for this thread
    influence_counts[nnz_idx] = window_size;

    // Store all influenced pixels
    int base_offset = nnz_idx * window_size * 2;  // Each pixel = 2 ints (row, col)
    int idx = 0;

    for (int out_i = out_i_start; out_i <= out_i_end; out_i++) {
        for (int out_j = out_j_start; out_j <= out_j_end; out_j++) {
            influenced_pixels[base_offset + idx * 2] = out_i;
            influenced_pixels[base_offset + idx * 2 + 1] = out_j;
            idx++;
        }
    }
}

// Helper structure for pixel coordinates
struct PixelCoord {
    int row, col;

    __host__ __device__
    bool operator==(const PixelCoord& other) const {
        return row == other.row && col == other.col;
    }

    __host__ __device__
    bool operator<(const PixelCoord& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

// Kernel 2: Compute Lanczos for each unique output pixel
__global__ void compute_lanczos_sparse_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    int input_rows,
    int input_cols,
    const int* __restrict__ output_pixel_list,  // [row, col] pairs
    int num_output_pixels,
    double scale_x,
    double scale_y,
    int kernel_size,
    double threshold,
    double* __restrict__ output_values,
    int* __restrict__ output_valid) {  // 1 if pixel has value > threshold

    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_output_pixels) return;

    int out_i = output_pixel_list[pixel_idx * 2];
    int out_j = output_pixel_list[pixel_idx * 2 + 1];

    // Map output position back to input space
    double orig_i = out_i / scale_y;
    double orig_j = out_j / scale_x;

    double sum = 0.0;
    double weight_sum = 0.0;

    // Lanczos window in input space
    int start_i = max(0, (int)floor(orig_i - kernel_size));
    int end_i = min(input_rows - 1, (int)ceil(orig_i + kernel_size));
    int start_j = max(0, (int)floor(orig_j - kernel_size));
    int end_j = min(input_cols - 1, (int)ceil(orig_j + kernel_size));

    // Iterate over input window
    for (int ii = start_i; ii <= end_i; ii++) {
        double dist_i = orig_i - ii;
        double weight_i = lanczos_kernel(dist_i, kernel_size);

        if (fabs(weight_i) < 1e-9) continue;

        // Search this row for nonzeros in column range
        int row_start = row_ptr[ii];
        int row_end = row_ptr[ii + 1];

        for (int idx = row_start; idx < row_end; idx++) {
            int jj = col_ind[idx];

            if (jj >= start_j && jj <= end_j) {
                double val = values[idx];
                double dist_j = orig_j - jj;
                double weight = weight_i * lanczos_kernel(dist_j, kernel_size);

                if (fabs(weight) > 1e-9) {
                    sum += val * weight;
                    weight_sum += weight;
                }
            }
        }
    }

    // Normalize and store result
    if (fabs(weight_sum) > 1e-9) {
        double result = sum / weight_sum;
        if (fabs(result) > threshold) {
            output_values[pixel_idx] = result;
            output_valid[pixel_idx] = 1;
        } else {
            output_values[pixel_idx] = 0.0;
            output_valid[pixel_idx] = 0;
        }
    } else {
        output_values[pixel_idx] = 0.0;
        output_valid[pixel_idx] = 0;
    }
}

// Host wrapper
void lanczos_sparse_gpu(
    const CSRDeviceMatrix& device_input,
    CSRMatrix& output,
    double scale_x, double scale_y,
    int kernel_size, double threshold) {

    std::cout << "Sparse Lanczos GPU: " << device_input.rows << "x" << device_input.cols
              << " -> scale " << scale_x << "x" << scale_y
              << ", kernel=" << kernel_size << std::endl;

    int new_rows = std::round(device_input.rows * scale_y);
    int new_cols = std::round(device_input.cols * scale_x);

    // Step 1: Compute maximum influence zone size
    int max_window_size = static_cast<int>(
        (2 * kernel_size * scale_y + 1) * (2 * kernel_size * scale_x + 1)
    );

    std::cout << "Max window size per nonzero: " << max_window_size << std::endl;

    // Allocate temporary storage for influenced pixels
    int* d_influenced_pixels;
    int* d_influence_counts;

    size_t influenced_size = device_input.nnz * max_window_size * 2 * sizeof(int);
    std::cout << "Allocating " << (influenced_size / 1e6) << " MB for influence zones" << std::endl;

    CUDA_CHECK(cudaMalloc(&d_influenced_pixels, influenced_size));
    CUDA_CHECK(cudaMalloc(&d_influence_counts, device_input.nnz * sizeof(int)));

    // Launch kernel 1: compute influence zones
    dim3 block_size(256);
    dim3 grid_size((device_input.nnz + block_size.x - 1) / block_size.x);

    compute_influence_zones_kernel<<<grid_size, block_size>>>(
        device_input.row_ptr, device_input.col_ind,
        device_input.rows, device_input.cols, device_input.nnz,
        scale_x, scale_y, kernel_size,
        new_rows, new_cols,
        d_influenced_pixels, d_influence_counts);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2: Flatten and deduplicate influenced pixels
    // First, compute prefix sum to know total number of influenced pixels
    thrust::device_ptr<int> d_counts_ptr(d_influence_counts);
    thrust::device_vector<int> d_offsets(device_input.nnz + 1);
    d_offsets[0] = 0;
    thrust::inclusive_scan(d_counts_ptr, d_counts_ptr + device_input.nnz, d_offsets.begin() + 1);

    int total_influenced;
    CUDA_CHECK(cudaMemcpy(&total_influenced,
                          thrust::raw_pointer_cast(d_offsets.data()) + device_input.nnz,
                          sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Total influenced pixels (with duplicates): " << total_influenced << std::endl;

    // Copy influenced pixels to contiguous array
    thrust::device_vector<int> d_pixel_coords(total_influenced * 2);

    // Compact the influenced pixels array
    // This is a bit tricky - we need to gather based on offsets
    // For simplicity, copy to host and process (can optimize later)

    std::vector<int> h_influence_counts(device_input.nnz);
    CUDA_CHECK(cudaMemcpy(h_influence_counts.data(), d_influence_counts,
                          device_input.nnz * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_influenced_pixels(total_influenced * 2);

    int write_pos = 0;
    for (int nnz_idx = 0; nnz_idx < device_input.nnz; nnz_idx++) {
        int count = h_influence_counts[nnz_idx];
        int read_pos = nnz_idx * max_window_size * 2;

        CUDA_CHECK(cudaMemcpy(h_influenced_pixels.data() + write_pos,
                              d_influenced_pixels + read_pos,
                              count * 2 * sizeof(int),
                              cudaMemcpyDeviceToHost));
        write_pos += count * 2;
    }

    CUDA_CHECK(cudaFree(d_influenced_pixels));
    CUDA_CHECK(cudaFree(d_influence_counts));

    // Step 3: Sort and deduplicate on host (can optimize with GPU later)
    std::vector<std::pair<int, int>> pixel_pairs;
    pixel_pairs.reserve(total_influenced);

    for (int i = 0; i < total_influenced; i++) {
        pixel_pairs.push_back({h_influenced_pixels[i * 2], h_influenced_pixels[i * 2 + 1]});
    }

    std::sort(pixel_pairs.begin(), pixel_pairs.end());
    auto last = std::unique(pixel_pairs.begin(), pixel_pairs.end());
    pixel_pairs.erase(last, pixel_pairs.end());

    int num_unique_pixels = pixel_pairs.size();
    std::cout << "Unique output pixels to compute: " << num_unique_pixels << std::endl;

    // Copy unique pixels back to device
    std::vector<int> h_unique_pixels(num_unique_pixels * 2);
    for (int i = 0; i < num_unique_pixels; i++) {
        h_unique_pixels[i * 2] = pixel_pairs[i].first;
        h_unique_pixels[i * 2 + 1] = pixel_pairs[i].second;
    }

    int* d_output_pixel_list;
    CUDA_CHECK(cudaMalloc(&d_output_pixel_list, num_unique_pixels * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_output_pixel_list, h_unique_pixels.data(),
                          num_unique_pixels * 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Step 4: Compute Lanczos for each unique pixel
    double* d_output_values;
    int* d_output_valid;

    CUDA_CHECK(cudaMalloc(&d_output_values, num_unique_pixels * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output_valid, num_unique_pixels * sizeof(int)));

    dim3 lanczos_grid((num_unique_pixels + 255) / 256);
    dim3 lanczos_block(256);

    compute_lanczos_sparse_kernel<<<lanczos_grid, lanczos_block>>>(
        device_input.row_ptr, device_input.col_ind, device_input.values,
        device_input.rows, device_input.cols,
        d_output_pixel_list, num_unique_pixels,
        scale_x, scale_y, kernel_size, threshold,
        d_output_values, d_output_valid);

    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 5: Copy results and compact
    std::vector<double> h_output_values(num_unique_pixels);
    std::vector<int> h_output_valid(num_unique_pixels);

    CUDA_CHECK(cudaMemcpy(h_output_values.data(), d_output_values,
                          num_unique_pixels * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_valid.data(), d_output_valid,
                          num_unique_pixels * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output_pixel_list));
    CUDA_CHECK(cudaFree(d_output_values));
    CUDA_CHECK(cudaFree(d_output_valid));

    // Build output CSR - validate coordinates first
    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;

    for (int i = 0; i < num_unique_pixels; i++) {
        if (h_output_valid[i]) {
            int row = h_unique_pixels[i * 2];
            int col = h_unique_pixels[i * 2 + 1];
            double val = h_output_values[i];

            // Validate coordinates
            if (row < 0 || row >= new_rows || col < 0 || col >= new_cols) {
                std::cerr << "Warning: Invalid coordinate (" << row << ", " << col
                          << ") for matrix " << new_rows << "x" << new_cols << std::endl;
                continue;
            }

            // Validate value
            if (!std::isfinite(val)) {
                std::cerr << "Warning: Non-finite value at (" << row << ", " << col << ")" << std::endl;
                continue;
            }

            out_rows.push_back(row);
            out_cols.push_back(col);
            out_vals.push_back(val);
        }
    }

    std::cout << "Final nonzeros: " << out_vals.size() << std::endl;

    if (out_vals.empty()) {
        std::cout << "Warning: No valid nonzeros produced, returning empty matrix" << std::endl;
        output.rows = new_rows;
        output.cols = new_cols;
        output.nnz = 0;
        output.row_ptr.assign(new_rows + 1, 0);
        output.col_ind.clear();
        output.values.clear();
        return;
    }

    output.rows = new_rows;
    output.cols = new_cols;
    output.nnz = out_vals.size();
    output.values = out_vals;
    output.col_ind = out_cols;

    // Build row_ptr
    output.row_ptr.assign(new_rows + 1, 0);

    for (size_t i = 0; i < out_rows.size(); i++) {
        if (out_rows[i] >= 0 && out_rows[i] < new_rows) {
            output.row_ptr[out_rows[i] + 1]++;
        }
    }

    // Prefix sum
    for (int i = 0; i < new_rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }

    // Verify row_ptr integrity
    if (output.row_ptr[new_rows] != output.nnz) {
        std::cerr << "ERROR: row_ptr[n_rows] = " << output.row_ptr[new_rows]
                  << " but nnz = " << output.nnz << std::endl;
    }

    std::cout << "Output: " << output.rows << "x" << output.cols
              << ", NNZ: " << output.nnz
              << " (density: " << (100.0 * output.nnz / (static_cast<double>(output.rows) * output.cols)) << "%)"
              << std::endl;
}