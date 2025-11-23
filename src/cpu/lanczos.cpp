#include "formats/matrix_formats.hpp"
#include "kernels/lanczos.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <omp.h>

struct Contribution {
    double value;
    double weight;
};

struct CoordHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

struct Triplet {
    int row;
    int col;
    double val;
    double weight;
};

CSRMatrix lanczos_cpu_reference(const CSRMatrix& input, double scale_x, double scale_y, int a = 3, double threshold = 1e-6) {
    int new_rows = std::round(input.rows * scale_y);
    int new_cols = std::round(input.cols * scale_x);

    std::vector<int> output_rows;
    std::vector<int> output_cols;
    std::vector<double> output_values;

    /* Pre-allocate based on input sparsity pattern */
    output_rows.reserve(input.nnz * 4);
    output_cols.reserve(input.nnz * 4);
    output_values.reserve(input.nnz * 4);

    for (int i = 0; i < new_rows; i++) {
        for (int j = 0; j < new_cols; j++) {
            double orig_i = i / scale_y;
            double orig_j = j / scale_x;

            double sum = 0.0;
            double weight_sum = 0.0;

            /* Lanczos window */
            int start_i = std::max(0, (int)std::floor(orig_i - a));
            int end_i = std::min(input.rows - 1, (int)std::ceil(orig_i + a));
            int start_j = std::max(0, (int)std::floor(orig_j - a));
            int end_j = std::min(input.cols - 1, (int)std::ceil(orig_j + a));

            for (int ii = start_i; ii <= end_i; ii++) {
                for (int jj = start_j; jj <= end_j; jj++) {
                    /* Check if this element exists in the sparse matrix */
                    bool found = false;
                    double val = 0.0;

                    /* Search in CSR format */
                    for (int idx = input.row_ptr[ii]; idx < input.row_ptr[ii + 1]; idx++) {
                        if (input.col_ind[idx] == jj) {
                            val = input.values[idx];
                            found = true;
                            break;
                        }
                    }

                    if (found) {
                        double dist_i = orig_i - ii;
                        double dist_j = orig_j - jj;
                        double weight = lanczos_kernel(dist_i, a) * lanczos_kernel(dist_j, a);

                        if (std::abs(weight) > 1e-9) {
                            sum += val * weight;
                            weight_sum += weight;
                        }
                    }
                }
            }

            if (std::abs(weight_sum) > 1e-9) {
                double result = sum / weight_sum;
                if (std::abs(result) > threshold) {
                    output_rows.push_back(i);
                    output_cols.push_back(j);
                    output_values.push_back(result);
                }
            }
        }
    }

    /* Build output CSR matrix */
    CSRMatrix output;
    output.rows = new_rows;
    output.cols = new_cols;
    output.nnz = output_values.size();
    output.values = output_values;
    output.col_ind = output_cols;

    /* Build row_ptr */
    output.row_ptr.resize(output.rows + 1, 0);
    for (size_t i = 0; i < output_rows.size(); i++) {
        output.row_ptr[output_rows[i] + 1]++;
    }

    /* Prefix sum */
    for (int i = 0; i < output.rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }

    return output;
}

/* Deterministic sparse Lanczos */
CSRMatrix lanczos_sparse_deterministic(
    const CSRMatrix& input,
    double scale_x, double scale_y,
    int kernel_size, double threshold) {

    int new_rows = std::round(input.rows * scale_y);
    int new_cols = std::round(input.cols * scale_x);

    // For each input nonzero, calculate which output pixels it affects
    std::unordered_map<std::pair<int,int>, std::vector<Contribution>, CoordHash> output_accumulator;

    // Forward pass: distribute input nonzeros to output
    for (int ii = 0; ii < input.rows; ii++) {
        for (int idx = input.row_ptr[ii]; idx < input.row_ptr[ii + 1]; idx++) {
            int jj = input.col_ind[idx];
            double input_val = input.values[idx];

            // This input pixel at (ii, jj) affects output pixels in a window
            // Calculate output window this input affects
            double out_i_center = ii * scale_y;
            double out_j_center = jj * scale_x;

            // Lanczos kernel has support of 'a' in INPUT space
            // In OUTPUT space, the support is 'a * scale'
            int out_i_start = std::max(0, (int)std::floor(out_i_center - kernel_size * scale_y));
            int out_i_end = std::min(new_rows - 1, (int)std::ceil(out_i_center + kernel_size * scale_y));
            int out_j_start = std::max(0, (int)std::floor(out_j_center - kernel_size * scale_x));
            int out_j_end = std::min(new_cols - 1, (int)std::ceil(out_j_center + kernel_size * scale_x));

            // Distribute this input pixel's contribution to all affected output pixels
            for (int out_i = out_i_start; out_i <= out_i_end; out_i++) {
                for (int out_j = out_j_start; out_j <= out_j_end; out_j++) {
                    // Map output position back to input space
                    double orig_i = out_i / scale_y;
                    double orig_j = out_j / scale_x;

                    // Calculate Lanczos weight
                    double dist_i = orig_i - ii;
                    double dist_j = orig_j - jj;

                    double weight = lanczos_kernel(dist_i, kernel_size) *
                                   lanczos_kernel(dist_j, kernel_size);

                    if (std::abs(weight) > 1e-9) {
                        output_accumulator[{out_i, out_j}].push_back({
                            input_val, weight
                        });
                    }
                }
            }
        }
    }

    // Normalize accumulated contributions
    std::vector<int> out_rows, out_cols;
    std::vector<double> out_vals;

    for (const auto& [pos, contributions] : output_accumulator) {
        double sum = 0.0;
        double weight_sum = 0.0;

        for (const auto& contrib : contributions) {
            sum += contrib.value * contrib.weight;
            weight_sum += contrib.weight;
        }

        if (std::abs(weight_sum) > 1e-9) {
            double result = sum / weight_sum;
            if (std::abs(result) > threshold) {
                out_rows.push_back(pos.first);
                out_cols.push_back(pos.second);
                out_vals.push_back(result);
            }
        }
    }

    /* Build output CSR matrix */
    CSRMatrix output;
    output.rows = new_rows;
    output.cols = new_cols;
    output.nnz = out_vals.size();
    output.values = out_vals;
    output.col_ind = out_cols;

    /* Build row_ptr */
    output.row_ptr.resize(output.rows + 1, 0);
    for (size_t i = 0; i < out_rows.size(); i++) {
        output.row_ptr[out_rows[i] + 1]++;
    }

    /* Prefix sum */
    for (int i = 0; i < output.rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }

    return output;
}

/* Parallel sparse Lanczos with per-thread accumulation and reduction */
CSRMatrix lanczos_sparse_omp(const CSRMatrix& input,
                                       double scale_x,
                                       double scale_y,
                                       int kernel_size,
                                       double threshold)
{
    int new_rows = std::round(input.rows * scale_y);
    int new_cols = std::round(input.cols * scale_x);

    std::vector<Triplet> global_triplets;

    #pragma omp parallel num_threads(8)
    {
        std::vector<Triplet> local;
        local.reserve(1024);

        #pragma omp for nowait schedule(dynamic)
        for (int ii = 0; ii < input.rows; ii++) {
            for (int idx = input.row_ptr[ii]; idx < input.row_ptr[ii + 1]; idx++) {
                int jj = input.col_ind[idx];
                double input_val = input.values[idx];

                double out_i_center = ii * scale_y;
                double out_j_center = jj * scale_x;

                int out_i_start = std::max(0, (int)std::floor(out_i_center - kernel_size * scale_y));
                int out_i_end   = std::min(new_rows - 1, (int)std::ceil(out_i_center + kernel_size * scale_y));
                int out_j_start = std::max(0, (int)std::floor(out_j_center - kernel_size * scale_x));
                int out_j_end   = std::min(new_cols - 1, (int)std::ceil(out_j_center + kernel_size * scale_x));

                for (int out_i = out_i_start; out_i <= out_i_end; out_i++) {
                    for (int out_j = out_j_start; out_j <= out_j_end; out_j++) {
                        double orig_i = out_i / scale_y;
                        double orig_j = out_j / scale_x;

                        double dist_i = orig_i - ii;
                        double dist_j = orig_j - jj;

                        double weight = lanczos_kernel(dist_i, kernel_size) *
                                        lanczos_kernel(dist_j, kernel_size);

                        if (std::abs(weight) > 1e-9) {
                            local.push_back({out_i, out_j, input_val, weight});
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            global_triplets.insert(global_triplets.end(),
                                   local.begin(), local.end());
        }
    }

    if (global_triplets.empty()) {
        CSRMatrix empty;
        empty.rows = new_rows;
        empty.cols = new_cols;
        empty.nnz = 0;
        empty.values = {};
        empty.col_ind = {};
        empty.row_ptr.assign(new_rows + 1, 0);
        return empty;
    }

    std::sort(global_triplets.begin(), global_triplets.end(),
              [](const Triplet& a, const Triplet& b) {
                  return std::tie(a.row, a.col) < std::tie(b.row, b.col);
              });

    std::vector<int> out_rows;
    std::vector<int> out_cols;
    std::vector<double> out_vals;

    out_rows.reserve(global_triplets.size());
    out_cols.reserve(global_triplets.size());
    out_vals.reserve(global_triplets.size());

    double acc_val = 0.0;
    double acc_weight = 0.0;
    int current_row = global_triplets[0].row;
    int current_col = global_triplets[0].col;

    for (size_t i = 0; i < global_triplets.size(); i++) {
        const auto& t = global_triplets[i];
        if (t.row != current_row || t.col != current_col) {
            double result = (std::abs(acc_weight) > 1e-9) ? acc_val / acc_weight : 0.0;
            if (std::abs(result) > threshold) {
                out_rows.push_back(current_row);
                out_cols.push_back(current_col);
                out_vals.push_back(result);
            }
            current_row = t.row;
            current_col = t.col;
            acc_val = t.val * t.weight;
            acc_weight = t.weight;
        } else {
            acc_val += t.val * t.weight;
            acc_weight += t.weight;
        }
    }

    // Last group
    double result = (std::abs(acc_weight) > 1e-9) ? acc_val / acc_weight : 0.0;
    if (std::abs(result) > threshold) {
        out_rows.push_back(current_row);
        out_cols.push_back(current_col);
        out_vals.push_back(result);
    }

    CSRMatrix output;
    output.rows = new_rows;
    output.cols = new_cols;
    output.nnz = out_vals.size();
    output.values = std::move(out_vals);
    output.col_ind = std::move(out_cols);
    output.row_ptr.assign(output.rows + 1, 0);

    for (size_t i = 0; i < out_rows.size(); i++) {
        output.row_ptr[out_rows[i] + 1]++;
    }
    for (int i = 0; i < output.rows; i++) {
        output.row_ptr[i + 1] += output.row_ptr[i];
    }

    return output;
}