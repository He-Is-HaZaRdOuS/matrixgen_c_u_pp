#pragma once
#include <vector>
#include <armadillo>

/* Lightweight CSR for I/O and GPU transfer */
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;

    /* Conversion to/from Armadillo for CPU computation */
    arma::sp_mat to_arma() const {
        arma::umat locations(2, nnz);
        arma::vec values_vec(nnz);

#pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            int start = row_ptr[i];
            int end   = row_ptr[i + 1];
            for (int j = start; j < end; j++) {
                locations(0, j) = i;
                locations(1, j) = col_ind[j];
                values_vec(j)   = values[j];
            }
        }

        return arma::sp_mat(locations, values_vec, rows, cols);
    }

    static CSRMatrix from_arma(const arma::sp_mat& mat) {
        CSRMatrix csr;
        csr.rows = mat.n_rows;
        csr.cols = mat.n_cols;
        csr.nnz  = mat.n_nonzero;

        csr.row_ptr.resize(csr.rows + 1, 0);
        csr.col_ind.resize(csr.nnz);
        csr.values.resize(csr.nnz);

        // First, count nonzeros per row
        std::vector<int> row_counts(csr.rows, 0);
#pragma omp parallel for
        for (int i = 0; i < csr.rows; i++) {
            int count = 0;
            for (arma::sp_mat::const_row_iterator it = mat.begin_row(i);
                 it != mat.end_row(i); ++it) {
                count++;
                 }
            row_counts[i] = count;
        }

        // Prefix sum to build row_ptr
        csr.row_ptr[0] = 0;
        for (int i = 1; i <= csr.rows; i++) {
            csr.row_ptr[i] = csr.row_ptr[i - 1] + row_counts[i - 1];
        }

        // Fill col_ind and values in parallel
#pragma omp parallel for
        for (int i = 0; i < csr.rows; i++) {
            int start = csr.row_ptr[i];
            int idx = 0;
            for (arma::sp_mat::const_row_iterator it = mat.begin_row(i);
                 it != mat.end_row(i); ++it) {
                int pos = start + idx;
                csr.col_ind[pos] = it.col();
                csr.values[pos]  = *it;
                idx++;
                 }
        }

        return csr;
    }

};