#include "kernels/lanczos.hpp"
#include "utils/cuda_utils.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <vector>
#include <iostream>
#include "formats/cuda_matrix_formats.cuh"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <cmath>
#include <algorithm>
#include <cassert>

// -------------------- kernels --------------------

// Kernel A: produce row_of_nnz array for O(1) mapping
__global__ void build_row_of_nnz_kernel(const int* __restrict__ row_ptr, int rows, int nnz, int* __restrict__ row_of_nnz) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int start = row_ptr[r];
    int end = row_ptr[r+1];
    for (int idx = start; idx < end; ++idx) {
        row_of_nnz[idx] = r;
    }
}

// Kernel B: compute per-nnz influence counts (number of output pixels influenced by this input nonzero)
__global__ void compute_influence_counts_kernel(
    const int* __restrict__ col_ind,
    const int* __restrict__ row_of_nnz,
    int input_rows, int input_cols,
    int total_nnz,
    double scale_x, double scale_y,
    int kernel_size,
    int output_rows, int output_cols,
    int* __restrict__ counts_out) {

    int nnz_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nnz_idx >= total_nnz) return;

    int ii = row_of_nnz[nnz_idx];
    int jj = col_ind[nnz_idx];

    double out_i_center = ii * scale_y;
    double out_j_center = jj * scale_x;

    int out_i_start = max(0, (int)floor(out_i_center - kernel_size * scale_y));
    int out_i_end   = min(output_rows - 1, (int)ceil(out_i_center + kernel_size * scale_y));
    int out_j_start = max(0, (int)floor(out_j_center - kernel_size * scale_x));
    int out_j_end   = min(output_cols - 1, (int)ceil(out_j_center + kernel_size * scale_x));

    int window_rows = out_i_end - out_i_start + 1;
    int window_cols = out_j_end - out_j_start + 1;
    int window_size = window_rows * window_cols;
    counts_out[nnz_idx] = window_size; // may be zero if outside
}

// Kernel C: write all influenced coordinates into contiguous device array using offsets
__global__ void write_influenced_coords_kernel(
    const int* __restrict__ col_ind,
    const int* __restrict__ row_of_nnz,
    int input_rows, int input_cols,
    int total_nnz,
    double scale_x, double scale_y,
    int kernel_size,
    int output_rows, int output_cols,
    const int* __restrict__ offsets,     // exclusive prefix sum per nnz (length total_nnz+1)
    int* __restrict__ coords_out) {      // coords_out is int[total_influenced * 2] flattened

    int nnz_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nnz_idx >= total_nnz) return;

    int ii = row_of_nnz[nnz_idx];
    int jj = col_ind[nnz_idx];

    double out_i_center = ii * scale_y;
    double out_j_center = jj * scale_x;

    int out_i_start = max(0, (int)floor(out_i_center - kernel_size * scale_y));
    int out_i_end   = min(output_rows - 1, (int)ceil(out_i_center + kernel_size * scale_y));
    int out_j_start = max(0, (int)floor(out_j_center - kernel_size * scale_x));
    int out_j_end   = min(output_cols - 1, (int)ceil(out_j_center + kernel_size * scale_x));

    int write_base = offsets[nnz_idx]; // exclusive prefix sum => write_base..write_base+counts-1
    int pos = 0;
    for (int out_i = out_i_start; out_i <= out_i_end; ++out_i) {
        for (int out_j = out_j_start; out_j <= out_j_end; ++out_j) {
            int idx = write_base + pos;
            coords_out[2*idx    ] = out_i;
            coords_out[2*idx + 1] = out_j;
            ++pos;
        }
    }
}

// Kernel D: evaluate lanczos for each unique pixel using CSR input
__global__ void compute_lanczos_for_pixels_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    int input_rows, int input_cols,
    const int64_t* __restrict__ keys,    // packed keys for unique pixels (row<<32 | col)
    int num_keys,
    double scale_x, double scale_y,
    int kernel_size,
    double threshold,
    double* __restrict__ out_vals,   // length num_keys
    int* __restrict__ out_valid) {   // 0/1 flags

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    uint32_t out_i = (uint32_t)(keys[idx] >> 32);
    uint32_t out_j = (uint32_t)(keys[idx] & 0xffffffffu);

    double orig_i = out_i / scale_y;
    double orig_j = out_j / scale_x;

    double sum = 0.0;
    double weight_sum = 0.0;

    int start_i = max(0, (int)floor(orig_i - kernel_size));
    int end_i   = min(input_rows - 1, (int)ceil(orig_i + kernel_size));
    int start_j = max(0, (int)floor(orig_j - kernel_size));
    int end_j   = min(input_cols - 1, (int)ceil(orig_j + kernel_size));

    for (int ii = start_i; ii <= end_i; ++ii) {
        double dist_i = orig_i - ii;
        double w_i = lanczos_kernel(dist_i, kernel_size);
        if (fabs(w_i) < 1e-12) continue;

        int rstart = row_ptr[ii];
        int rend   = row_ptr[ii+1];

        // iterate nonzeros in row ii
        for (int k = rstart; k < rend; ++k) {
            int jj = col_ind[k];
            if (jj < start_j || jj > end_j) continue;
            double val = values[k];
            double dist_j = orig_j - jj;
            double w = w_i * lanczos_kernel(dist_j, kernel_size);
            if (fabs(w) < 1e-12) continue;
            sum += val * w;
            weight_sum += w;
        }
    }

    if (fabs(weight_sum) > 1e-12) {
        double res = sum / weight_sum;
        if (fabs(res) > threshold && isfinite(res)) {
            out_vals[idx] = res;
            out_valid[idx] = 1;
        } else {
            out_vals[idx] = 0.0;
            out_valid[idx] = 0;
        }
    } else {
        out_vals[idx] = 0.0;
        out_valid[idx] = 0;
    }
}

// -------------------- host pipeline --------------------

void lanczos_sparse_gpu_improved(
    const CSRDeviceMatrix& device_input,
    CSRMatrix& output,               // host output
    double scale_x, double scale_y,
    int kernel_size, double threshold) {

    std::cout << "Improved Sparse Lanczos GPU: " << device_input.rows << "x" << device_input.cols
              << " -> scale " << scale_x << "x" << scale_y << ", kernel=" << kernel_size << std::endl;

    int out_rows = (int)std::round(device_input.rows * scale_y);
    int out_cols = (int)std::round(device_input.cols * scale_x);
    int nnz = device_input.nnz;

    // 1) Build row_of_nnz array on device
    thrust::device_vector<int> d_row_of_nnz(nnz);
    {
        int block = 256;
        int grid = (device_input.rows + block - 1) / block;
        build_row_of_nnz_kernel<<<grid, block>>>(
            device_input.row_ptr, device_input.rows, nnz, thrust::raw_pointer_cast(d_row_of_nnz.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 2) Compute counts per nnz
    thrust::device_vector<int> d_counts(nnz);
    {
        int block = 256;
        int grid = (nnz + block - 1) / block;
        compute_influence_counts_kernel<<<grid, block>>>(
            device_input.col_ind,
            thrust::raw_pointer_cast(d_row_of_nnz.data()),
            device_input.rows, device_input.cols,
            nnz,
            scale_x, scale_y,
            kernel_size,
            out_rows, out_cols,
            thrust::raw_pointer_cast(d_counts.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 3) Exclusive scan -> offsets (length nnz+1)
    thrust::device_vector<int> d_offsets(nnz + 1);
    d_offsets[0] = 0;
    thrust::exclusive_scan(thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin() + 1);
    int total_influenced;
    CUDA_CHECK(cudaMemcpy(&total_influenced, thrust::raw_pointer_cast(d_offsets.data()) + nnz, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Total influenced pixels (with duplicates): " << total_influenced << std::endl;
    if (total_influenced == 0) {
        std::cout << "No influenced pixels -> empty output" << std::endl;
        output.rows = out_rows; output.cols = out_cols; output.nnz = 0;
        output.row_ptr.assign(out_rows + 1, 0);
        output.col_ind.clear(); output.values.clear();
        return;
    }

    // 4) Allocate coords buffer on device and write coords
    thrust::device_vector<int> d_coords(total_influenced * 2); // pair (row,col)
    {
        int block = 256;
        int grid = (nnz + block - 1) / block;
        write_influenced_coords_kernel<<<grid, block>>>(
            device_input.col_ind,
            thrust::raw_pointer_cast(d_row_of_nnz.data()),
            device_input.rows, device_input.cols,
            nnz,
            scale_x, scale_y,
            kernel_size,
            out_rows, out_cols,
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_coords.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 5) Pack coords into 64-bit keys: key = ((uint64_t)row << 32) | col
    thrust::device_vector<uint64_t> d_keys(total_influenced);
    {
        // kernel to pack keys
        const int block = 256;
        const int grid = (total_influenced + block - 1) / block;
        auto pack_kernel = [] __device__ (const int* coords, uint64_t* keys, int n) {
            // not allowed to use lambda with __global__ easily; we'll do a simple kernel below
            (void)coords; (void)keys; (void)n;
        };
        // simple launch via explicit kernel below
        // define separate kernel here:
    }

    // explicit kernel for packing keys
    auto pack_keys_kernel = [] __global__ (const int* coords, uint64_t* keys, int total) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;
        uint32_t r = (uint32_t)coords[2*idx];
        uint32_t c = (uint32_t)coords[2*idx + 1];
        uint64_t k = ( (uint64_t)r << 32 ) | (uint64_t)c;
        keys[idx] = k;
    };

    // Launch pack kernel (need to wrap as a named kernel)
    // Workaround: declare a real kernel function below and call it.
    // So remove lambda usage. We'll declare pack_keys as real kernel outside.

    // ---- declare pack_keys kernel (actual) ----
    // (declared below; we now call it)

    // call pack_keys
    {
        extern __global__ void pack_keys_kernel_glob(const int* coords, uint64_t* keys, int total);
        int block = 256;
        int grid = (total_influenced + block - 1) / block;
        pack_keys_kernel_glob<<<grid, block>>>(
            thrust::raw_pointer_cast(d_coords.data()), thrust::raw_pointer_cast(d_keys.data()), total_influenced);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 6) Sort keys and unique
    thrust::sort(thrust::device, d_keys.begin(), d_keys.end());
    auto new_end = thrust::unique(thrust::device, d_keys.begin(), d_keys.end());
    int num_unique = (int)(new_end - d_keys.begin());
    std::cout << "Unique output pixels to compute: " << num_unique << std::endl;

    // shrink vector to num_unique
    d_keys.resize(num_unique);

    // 7) Compute Lanczos values per unique pixel
    thrust::device_vector<double> d_out_vals(num_unique);
    thrust::device_vector<int> d_out_valid(num_unique);
    {
        int block = 256;
        int grid = (num_unique + block - 1) / block;
        compute_lanczos_for_pixels_kernel<<<grid, block>>>(
            device_input.row_ptr, device_input.col_ind, device_input.values,
            device_input.rows, device_input.cols,
            (const int64_t*)thrust::raw_pointer_cast(d_keys.data()),
            num_unique,
            scale_x, scale_y,
            kernel_size,
            threshold,
            thrust::raw_pointer_cast(d_out_vals.data()),
            thrust::raw_pointer_cast(d_out_valid.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 8) Compact results: create array of (key, val) for valid==1
    // We'll copy d_out_valid to a temporary vector of indices via stable_partition-like approach

    // Build indices [0..num_unique)
    thrust::device_vector<int> d_idx(num_unique);
    thrust::sequence(thrust::device, d_idx.begin(), d_idx.end());

    // Use thrust::copy_if to select valid indices
    thrust::device_vector<int> d_valid_idx(num_unique);
    auto end_it = thrust::copy_if(
        thrust::device,
        d_idx.begin(), d_idx.end(),
        d_out_valid.begin(),
        d_valid_idx.begin(),
        [] __device__ (int v) { return v != 0; }
    );
    int num_valid = (int)(end_it - d_valid_idx.begin());
    std::cout << "Final nonzeros: " << num_valid << std::endl;

    if (num_valid == 0) {
        // empty output
        output.rows = out_rows; output.cols = out_cols; output.nnz = 0;
        output.row_ptr.assign(out_rows + 1, 0);
        output.col_ind.clear(); output.values.clear();
        return;
    }

    // Gather keys and values for valid indices
    thrust::device_vector<uint64_t> d_final_keys(num_valid);
    thrust::device_vector<double> d_final_vals(num_valid);
    {
        // gather keys
        thrust::gather(thrust::device, d_valid_idx.begin(), d_valid_idx.begin() + num_valid, d_keys.begin(), d_final_keys.begin());
        // gather vals
        thrust::gather(thrust::device, d_valid_idx.begin(), d_valid_idx.begin() + num_valid, d_out_vals.begin(), d_final_vals.begin());
    }

    // 9) Convert keys -> row,col arrays on device
    thrust::device_vector<int> d_final_rows(num_valid);
    thrust::device_vector<int> d_final_cols(num_valid);
    {
        int block = 256;
        int grid = (num_valid + block - 1) / block;
        extern __global__ void unpack_keys_kernel_glob(const uint64_t* keys, int* rows, int* cols, int n);
        unpack_keys_kernel_glob<<<grid, block>>>(
            thrust::raw_pointer_cast(d_final_keys.data()),
            thrust::raw_pointer_cast(d_final_rows.data()),
            thrust::raw_pointer_cast(d_final_cols.data()),
            num_valid);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 10) Copy final coords and values to host
    std::vector<int> h_rows(num_valid), h_cols(num_valid);
    std::vector<double> h_vals(num_valid);
    CUDA_CHECK(cudaMemcpy(h_rows.data(), thrust::raw_pointer_cast(d_final_rows.data()), sizeof(int) * num_valid, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cols.data(), thrust::raw_pointer_cast(d_final_cols.data()), sizeof(int) * num_valid, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vals.data(), thrust::raw_pointer_cast(d_final_vals.data()), sizeof(double) * num_valid, cudaMemcpyDeviceToHost));

    // 11) Build host CSR
    output.rows = out_rows;
    output.cols = out_cols;
    output.nnz = num_valid;
    output.values = std::move(h_vals);
    output.col_ind = std::move(h_cols);
    output.row_ptr.assign(out_rows + 1, 0);
    for (int i = 0; i < num_valid; ++i) {
        int r = h_rows[i];
        if (r >= 0 && r < out_rows) output.row_ptr[r + 1]++;
    }
    // prefix sum
    for (int i = 0; i < out_rows; ++i) output.row_ptr[i + 1] += output.row_ptr[i];

    // Sanity check
    if ((size_t)output.row_ptr[out_rows] != output.nnz) {
        // If ordering not row-sorted, we must reorder entries by row to build CSR correctly.
        // Quick fix: create vector of triplets and stable sort by row then col
        struct Trip { int r,c; double v; };
        std::vector<Trip> trip(output.nnz);
        for (int i = 0; i < output.nnz; ++i) trip[i] = { h_rows[i], h_cols[i], output.values[i] };
        std::stable_sort(trip.begin(), trip.end(), [](const Trip&a,const Trip&b){ if (a.r!=b.r) return a.r<b.r; return a.c<b.c;});
        // rebuild arrays
        output.row_ptr.assign(out_rows + 1, 0);
        for (int i = 0; i < (int)trip.size(); ++i) {
            output.row_ptr[trip[i].r + 1]++;
            output.col_ind[i] = trip[i].c;
            output.values[i] = trip[i].v;
        }
        for (int i = 0; i < out_rows; ++i) output.row_ptr[i + 1] += output.row_ptr[i];
    }

    std::cout << "Output: " << output.rows << "x" << output.cols << ", NNZ: " << output.nnz << std::endl;
}

// -------------------- small kernels declared/defined below --------------------

// kernel to pack keys (coords -> uint64 key)
__global__ void pack_keys_kernel_glob(const int* coords, uint64_t* keys, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint32_t r = (uint32_t)coords[2*idx];
    uint32_t c = (uint32_t)coords[2*idx + 1];
    uint64_t k = ( (uint64_t)r << 32 ) | (uint64_t)c;
    keys[idx] = k;
}

// kernel to unpack keys to row/col
__global__ void unpack_keys_kernel_glob(const uint64_t* keys, int* rows, int* cols, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t k = keys[idx];
    rows[idx] = (int)(k >> 32);
    cols[idx] = (int)(k & 0xffffffffu);
}

