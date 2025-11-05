#include "formats/matrix_formats.hpp"
#include "kernels/lanczos.hpp"
#include <cmath>
#include <iostream>

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