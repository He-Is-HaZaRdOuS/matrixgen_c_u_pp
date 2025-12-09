/*
 * This is a good balance between speed and efficiency, uses chunking strategy to not overflow vram during densification
*/
#include <algorithm>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

#include "formats/cuda_matrix_formats.cuh"
#include "kernels/lanczos.hpp"
#include "utils/cuda_utils.cuh"

__global__ void build_row_of_nnz_kernel_c(const int* __restrict__ row_ptr,
                                        int rows, int nnz,
                                        int* __restrict__ row_of_nnz)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int start = row_ptr[r];
    int end   = row_ptr[r + 1];
    for (int idx = start; idx < end; ++idx) {
        row_of_nnz[idx] = r;
    }
}

__global__ void compute_influence_counts_kernel(
    const int* __restrict__ col_ind,
    const int* __restrict__ row_of_nnz,
    int input_rows, int input_cols,
    int nnz_start, int nnz_count,
    double scale_x, double scale_y,
    int kernel_size,
    int output_rows, int output_cols,
    int* __restrict__ counts_out)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= nnz_count) return;

    int nnz_idx = nnz_start + local_idx;

    int ii = row_of_nnz[nnz_idx];
    int jj = col_ind[nnz_idx];

    double out_i_center = ii * scale_y;
    double out_j_center = jj * scale_x;

    int out_i_start = max(0, (int)floor(out_i_center - kernel_size * scale_y));
    int out_i_end   = min(output_rows - 1, (int)ceil(out_i_center + kernel_size * scale_y));
    int out_j_start = max(0, (int)floor(out_j_center - kernel_size * scale_x));
    int out_j_end   = min(output_cols - 1, (int)ceil(out_j_center + kernel_size * scale_x));

    if (out_i_end < out_i_start || out_j_end < out_j_start) {
        counts_out[local_idx] = 0;
        return;
    }

    int window_rows = out_i_end - out_i_start + 1;
    int window_cols = out_j_end - out_j_start + 1;
    counts_out[local_idx] = window_rows * window_cols;
}

__global__ void write_contribs_kernel(
    const int* __restrict__ col_ind,
    const int* __restrict__ row_of_nnz,
    const double* __restrict__ values,
    int input_rows, int input_cols,
    int nnz_start, int nnz_count,
    double scale_x, double scale_y,
    int kernel_size,
    int output_rows, int output_cols,
    const int* __restrict__ offsets,
    uint64_t* __restrict__ keys_out,
    double* __restrict__ num_out,
    double* __restrict__ den_out)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= nnz_count) return;

    int nnz_idx   = nnz_start + local_idx;
    int write_base = offsets[local_idx];
    int ii = row_of_nnz[nnz_idx];
    int jj = col_ind[nnz_idx];
    double val = values[nnz_idx];

    double out_i_center = ii * scale_y;
    double out_j_center = jj * scale_x;

    int out_i_start = max(0, (int)floor(out_i_center - kernel_size * scale_y));
    int out_i_end   = min(output_rows - 1, (int)ceil(out_i_center + kernel_size * scale_y));
    int out_j_start = max(0, (int)floor(out_j_center - kernel_size * scale_x));
    int out_j_end   = min(output_cols - 1, (int)ceil(out_j_center + kernel_size * scale_x));

    if (out_i_end < out_i_start || out_j_end < out_j_start) {
        return;
    }

    int pos = 0;

    for (int out_i = out_i_start; out_i <= out_i_end; ++out_i) {
        double orig_i = out_i / scale_y;
        double dist_i = orig_i - ii;
        double w_i = lanczos_kernel(dist_i, kernel_size);
        if (fabs(w_i) < 1e-12) {
            pos += (out_j_end - out_j_start + 1);
            continue;
        }

        for (int out_j = out_j_start; out_j <= out_j_end; ++out_j) {
            double orig_j = out_j / scale_x;
            double dist_j = orig_j - jj;
            double w_j = lanczos_kernel(dist_j, kernel_size);
            double w = w_i * w_j;

            int idx = write_base + pos;
            uint32_t r = (uint32_t)out_i;
            uint32_t c = (uint32_t)out_j;
            keys_out[idx] = ( (uint64_t)r << 32 ) | (uint64_t)c;

            if (fabs(w) < 1e-12) {
                num_out[idx] = 0.0;
                den_out[idx] = 0.0;
            } else {
                num_out[idx] = val * w;
                den_out[idx] = w;
            }
            ++pos;
        }
    }
}

void lanczos_sparse_gpu_improved_chunked(
    const CSRDeviceMatrix& device_input,
    CSRMatrix& output,               // host output
    double scale_x, double scale_y,
    int kernel_size, double threshold)
{
    std::cout << "Improved Sparse Lanczos GPU (chunked): "
              << device_input.rows << "x" << device_input.cols
              << " -> scale " << scale_x << "x" << scale_y
              << ", kernel=" << kernel_size << std::endl;

    int out_rows = (int)std::round(device_input.rows * scale_y);
    int out_cols = (int)std::round(device_input.cols * scale_x);
    int nnz      = device_input.nnz;

    if (nnz == 0) {
        output.rows = out_rows;
        output.cols = out_cols;
        output.nnz  = 0;
        output.row_ptr.assign(out_rows + 1, 0);
        output.col_ind.clear();
        output.values.clear();
        std::cout << "Empty input matrix -> empty output\n";
        return;
    }

    thrust::device_vector<int> d_row_of_nnz(nnz);
    {
        int block = 256;
        int grid  = (device_input.rows + block - 1) / block;
        build_row_of_nnz_kernel_c<<<grid, block>>>(
            device_input.row_ptr, device_input.rows, nnz,
            thrust::raw_pointer_cast(d_row_of_nnz.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }



    thrust::device_vector<uint64_t> d_global_keys;
    thrust::device_vector<double>   d_global_num;
    thrust::device_vector<double>   d_global_den;

    thrust::device_vector<int> d_counts;
    thrust::device_vector<int> d_offsets;

    thrust::device_vector<int> d_counts_all(nnz);
    {
        int block = 256;
        int grid = (nnz + block - 1) / block;
        compute_influence_counts_kernel<<<grid, block>>>(
            device_input.col_ind,
            thrust::raw_pointer_cast(d_row_of_nnz.data()),
            device_input.rows, device_input.cols,
            0, nnz,
            scale_x, scale_y,
            kernel_size,
            out_rows, out_cols,
            thrust::raw_pointer_cast(d_counts_all.data()));
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    thrust::device_vector<long long> d_scan(nnz + 1);
    d_scan[0] = 0;
    thrust::exclusive_scan(thrust::device, d_counts_all.begin(), d_counts_all.end(), d_scan.begin() + 1);
    long long total_influenced_all = 0;
    CUDA_CHECK(cudaMemcpy(&total_influenced_all,
                          thrust::raw_pointer_cast(d_scan.data()) + nnz,
                          sizeof(long long),
                          cudaMemcpyDeviceToHost));
    std::cout << "Total influenced (global): " << total_influenced_all << std::endl;

    if (total_influenced_all == 0) { /* empty result early return as before */ }

    /* vram budget heuristic */
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t budget = static_cast<size_t>(free_mem * 0.45);
    const size_t bytes_per_influence = 24;
    size_t max_influences_allowed = budget / bytes_per_influence;
    if (max_influences_allowed < 8192) max_influences_allowed = 8192;

    double avg_infl_per_nnz = (double)total_influenced_all / (double)nnz;
    size_t nnz_per_chunk_guess = (size_t)std::max( (double)1.0, (double)max_influences_allowed / std::max(1.0, avg_infl_per_nnz) );

    /* Hard upper limit */
    const size_t HARD_CAP_NNZ_PER_CHUNK = 200000; /* 200k nnz per chunk max */
    const size_t MIN_NNZ_PER_CHUNK = 256;

    size_t nnz_per_chunk = std::min( HARD_CAP_NNZ_PER_CHUNK, std::max(MIN_NNZ_PER_CHUNK, nnz_per_chunk_guess) );
    if (nnz_per_chunk > (size_t)nnz) nnz_per_chunk = nnz;

    std::cout << "Memory free: " << (free_mem / (1024.0*1024.0)) << " MB, "
              << "total_influenced_all: " << total_influenced_all << ", "
              << "avg_infl/nz: " << avg_infl_per_nnz << ", "
              << "nnz_per_chunk initial: " << nnz_per_chunk << std::endl;

    int processed_nnz = 0;
    while (processed_nnz < nnz) {
        int chunk_start = processed_nnz;
        int remaining   = nnz - processed_nnz;
        int chunk_n     = (int)std::min((size_t)remaining, nnz_per_chunk);

        int scan_before = 0, scan_after = 0;
        CUDA_CHECK(cudaMemcpy(&scan_before, thrust::raw_pointer_cast(d_scan.data()) + chunk_start, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&scan_after,  thrust::raw_pointer_cast(d_scan.data()) + (chunk_start + chunk_n), sizeof(int), cudaMemcpyDeviceToHost));
        int total_influenced_chunk = scan_after - scan_before;

        while ((size_t)total_influenced_chunk > max_influences_allowed && chunk_n > (int)MIN_NNZ_PER_CHUNK) {
            chunk_n = std::max((int)MIN_NNZ_PER_CHUNK, chunk_n / 2);
            CUDA_CHECK(cudaMemcpy(&scan_after, thrust::raw_pointer_cast(d_scan.data()) + (chunk_start + chunk_n), sizeof(int), cudaMemcpyDeviceToHost));
            total_influenced_chunk = scan_after - scan_before;
        }
        if (total_influenced_chunk == 0) {
            processed_nnz += chunk_n;
            continue;
        }

        std::cout << "Processing NNZ chunk: [" << chunk_start << "," << (chunk_start + chunk_n) << "), influences: " << total_influenced_chunk << std::endl;

        thrust::device_vector<int> d_counts_chunk(chunk_n);
        thrust::device_vector<int> d_offsets_chunk(chunk_n + 1);

        CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_counts_chunk.data()),
                              thrust::raw_pointer_cast(d_counts_all.data()) + chunk_start,
                              chunk_n * sizeof(int), cudaMemcpyDeviceToDevice));

        d_offsets_chunk[0] = 0;
        thrust::exclusive_scan(thrust::device, d_counts_chunk.begin(), d_counts_chunk.end(), d_offsets_chunk.begin() + 1);
        int total_influenced_chunk_check = 0;
        CUDA_CHECK(cudaMemcpy(&total_influenced_chunk_check, thrust::raw_pointer_cast(d_offsets_chunk.data()) + chunk_n, sizeof(int), cudaMemcpyDeviceToHost));
        /* Sanity check */
        if (total_influenced_chunk_check != total_influenced_chunk) {
            int block = 256;
            int grid  = (chunk_n + block - 1) / block;
            compute_influence_counts_kernel<<<grid, block>>>(
                device_input.col_ind,
                thrust::raw_pointer_cast(d_row_of_nnz.data()),
                device_input.rows, device_input.cols,
                chunk_start, chunk_n,
                scale_x, scale_y,
                kernel_size,
                out_rows, out_cols,
                thrust::raw_pointer_cast(d_counts_chunk.data()));
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaDeviceSynchronize());
            d_offsets_chunk[0] = 0;
            thrust::exclusive_scan(thrust::device, d_counts_chunk.begin(), d_counts_chunk.end(), d_offsets_chunk.begin() + 1);
            CUDA_CHECK(cudaMemcpy(&total_influenced_chunk_check, thrust::raw_pointer_cast(d_offsets_chunk.data()) + chunk_n, sizeof(int), cudaMemcpyDeviceToHost));
            total_influenced_chunk = total_influenced_chunk_check;
        }

        thrust::device_vector<uint64_t> d_keys_chunk(total_influenced_chunk);
        thrust::device_vector<double>   d_num_chunk(total_influenced_chunk);
        thrust::device_vector<double>   d_den_chunk(total_influenced_chunk);

        {
            int block = 256;
            int grid  = (chunk_n + block - 1) / block;
            write_contribs_kernel<<<grid, block>>>(
                device_input.col_ind,
                thrust::raw_pointer_cast(d_row_of_nnz.data()),
                device_input.values,
                device_input.rows, device_input.cols,
                chunk_start, chunk_n,
                scale_x, scale_y,
                kernel_size,
                out_rows, out_cols,
                thrust::raw_pointer_cast(d_offsets_chunk.data()),
                thrust::raw_pointer_cast(d_keys_chunk.data()),
                thrust::raw_pointer_cast(d_num_chunk.data()),
                thrust::raw_pointer_cast(d_den_chunk.data()));
            CUDA_CHECK_LAST_ERROR();
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        auto zip_in_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_num_chunk.begin(), d_den_chunk.begin()));
        auto zip_in_end   = thrust::make_zip_iterator(
            thrust::make_tuple(d_num_chunk.end(),   d_den_chunk.end()));

        thrust::sort_by_key(d_keys_chunk.begin(), d_keys_chunk.end(), zip_in_begin);

        thrust::device_vector<uint64_t> d_keys_chunk_red(total_influenced_chunk);
        thrust::device_vector<double>   d_num_chunk_red(total_influenced_chunk);
        thrust::device_vector<double>   d_den_chunk_red(total_influenced_chunk);

        auto zip_out_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_num_chunk_red.begin(), d_den_chunk_red.begin()));

        thrust::equal_to<uint64_t> key_eq;
        auto sum_zip = [] __device__ (const thrust::tuple<double,double>& a,
                                      const thrust::tuple<double,double>& b) {
            return thrust::make_tuple(
                thrust::get<0>(a) + thrust::get<0>(b),
                thrust::get<1>(a) + thrust::get<1>(b));
        };

        auto red_end = thrust::reduce_by_key(
            d_keys_chunk.begin(), d_keys_chunk.end(),
            zip_in_begin,
            d_keys_chunk_red.begin(),
            zip_out_begin,
            key_eq, sum_zip);

        int chunk_unique = (int)(red_end.first - d_keys_chunk_red.begin());
        d_keys_chunk_red.resize(chunk_unique);
        d_num_chunk_red.resize(chunk_unique);
        d_den_chunk_red.resize(chunk_unique);

        std::cout << "  chunk unique pixels: " << chunk_unique << std::endl;
        if (chunk_unique == 0) {
            continue;
        }

        size_t old_size = d_global_keys.size();
        d_global_keys.resize(old_size + chunk_unique);
        d_global_num.resize(old_size + chunk_unique);
        d_global_den.resize(old_size + chunk_unique);

        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_global_keys.data()) + old_size,
            thrust::raw_pointer_cast(d_keys_chunk_red.data()),
            chunk_unique * sizeof(uint64_t),
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_global_num.data()) + old_size,
            thrust::raw_pointer_cast(d_num_chunk_red.data()),
            chunk_unique * sizeof(double),
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(d_global_den.data()) + old_size,
            thrust::raw_pointer_cast(d_den_chunk_red.data()),
            chunk_unique * sizeof(double),
            cudaMemcpyDeviceToDevice));

        processed_nnz += chunk_n;
    }

    if (d_global_keys.empty()) {
        std::cout << "No nonzero outputs after all chunks\n";
        output.rows = out_rows;
        output.cols = out_cols;
        output.nnz  = 0;
        output.row_ptr.assign(out_rows + 1, 0);
        output.col_ind.clear();
        output.values.clear();
        return;
    }

    {
        auto zip_in_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_global_num.begin(), d_global_den.begin()));
        auto zip_in_end   = thrust::make_zip_iterator(
            thrust::make_tuple(d_global_num.end(),   d_global_den.end()));

        thrust::sort_by_key(d_global_keys.begin(), d_global_keys.end(), zip_in_begin);

        thrust::device_vector<uint64_t> d_final_keys(d_global_keys.size());
        thrust::device_vector<double>   d_final_num(d_global_keys.size());
        thrust::device_vector<double>   d_final_den(d_global_keys.size());

        auto zip_out_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_final_num.begin(), d_final_den.begin()));

        thrust::equal_to<uint64_t> key_eq;
        auto sum_zip = [] __device__ (const thrust::tuple<double,double>& a,
                                      const thrust::tuple<double,double>& b) {
            return thrust::make_tuple(
                thrust::get<0>(a) + thrust::get<0>(b),
                thrust::get<1>(a) + thrust::get<1>(b));
        };

        auto red_end = thrust::reduce_by_key(
            d_global_keys.begin(), d_global_keys.end(),
            zip_in_begin,
            d_final_keys.begin(),
            zip_out_begin,
            key_eq, sum_zip);

        int num_final = (int)(red_end.first - d_final_keys.begin());
        d_final_keys.resize(num_final);
        d_final_num.resize(num_final);
        d_final_den.resize(num_final);

        std::cout << "Global unique pixels after merge: " << num_final << std::endl;

        if (num_final == 0) {
            output.rows = out_rows;
            output.cols = out_cols;
            output.nnz  = 0;
            output.row_ptr.assign(out_rows + 1, 0);
            output.col_ind.clear();
            output.values.clear();
            return;
        }


        std::vector<uint64_t> h_keys(num_final);
        std::vector<double>   h_num(num_final);
        std::vector<double>   h_den(num_final);

        CUDA_CHECK(cudaMemcpy(h_keys.data(), thrust::raw_pointer_cast(d_final_keys.data()),
                              num_final * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_num.data(),  thrust::raw_pointer_cast(d_final_num.data()),
                              num_final * sizeof(double),   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_den.data(),  thrust::raw_pointer_cast(d_final_den.data()),
                              num_final * sizeof(double),   cudaMemcpyDeviceToHost));

        std::vector<int>    h_rows;
        std::vector<int>    h_cols;
        std::vector<double> h_vals;
        h_rows.reserve(num_final);
        h_cols.reserve(num_final);
        h_vals.reserve(num_final);

        for (int i = 0; i < num_final; ++i) {
            double denom = h_den[i];
            if (fabs(denom) < 1e-12) continue;
            double v = h_num[i] / denom;
            if (!std::isfinite(v)) continue;
            if (fabs(v) <= threshold) continue;

            uint64_t k = h_keys[i];
            int r = (int)(k >> 32);
            int c = (int)(k & 0xffffffffu);
            if (r < 0 || r >= out_rows || c < 0 || c >= out_cols) continue;

            h_rows.push_back(r);
            h_cols.push_back(c);
            h_vals.push_back(v);
        }

        int final_nnz = (int)h_rows.size();
        std::cout << "Final nonzeros after threshold: " << final_nnz << std::endl;

        /* Build CSR on host from COO */
        output.rows = out_rows;
        output.cols = out_cols;
        output.nnz  = final_nnz;
        output.values = h_vals;
        output.col_ind = h_cols;
        output.row_ptr.assign(out_rows + 1, 0);

        for (int i = 0; i < final_nnz; ++i) {
            int r = h_rows[i];
            if (r >= 0 && r < out_rows)
                output.row_ptr[r + 1]++;
        }
        for (int r = 0; r < out_rows; ++r) {
            output.row_ptr[r + 1] += output.row_ptr[r];
        }

        if (output.row_ptr[out_rows] != final_nnz) {
            struct Trip { int r, c; double v; };
            std::vector<Trip> trip(final_nnz);
            for (int i = 0; i < final_nnz; ++i)
                trip[i] = { h_rows[i], h_cols[i], h_vals[i] };
            std::stable_sort(trip.begin(), trip.end(),
                [](const Trip& a, const Trip& b) {
                    if (a.r != b.r) return a.r < b.r;
                    return a.c < b.c;
                });

            output.row_ptr.assign(out_rows + 1, 0);
            output.col_ind.resize(final_nnz);
            output.values.resize(final_nnz);
            for (int i = 0; i < final_nnz; ++i) {
                output.row_ptr[trip[i].r + 1]++;
                output.col_ind[i] = trip[i].c;
                output.values[i]  = trip[i].v;
            }
            for (int r = 0; r < out_rows; ++r)
                output.row_ptr[r + 1] += output.row_ptr[r];
        }

        std::cout << "Output: " << output.rows << "x" << output.cols
                  << ", NNZ: " << output.nnz << std::endl;
    }
}
