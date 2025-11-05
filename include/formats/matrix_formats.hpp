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

        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                locations(0, idx) = i;
                locations(1, idx) = col_ind[j];
                values_vec(idx) = values[j];
                idx++;
            }
        }

        return arma::sp_mat(locations, values_vec, rows, cols);
    }

    static CSRMatrix from_arma(const arma::sp_mat& mat) {
        CSRMatrix csr;
        csr.rows = mat.n_rows;
        csr.cols = mat.n_cols;
        csr.nnz = mat.n_nonzero;

        csr.row_ptr.resize(csr.rows + 1, 0);
        csr.col_ind.reserve(csr.nnz);
        csr.values.reserve(csr.nnz);

        /* Armadillo provides efficient iteration */
        for (int i = 0; i < csr.rows; i++) {
            csr.row_ptr[i] = csr.col_ind.size();
            for (arma::sp_mat::const_row_iterator it = mat.begin_row(i);
                 it != mat.end_row(i); ++it) {
                csr.col_ind.push_back(it.col());
                csr.values.push_back(*it);
                 }
        }
        csr.row_ptr[csr.rows] = csr.nnz;

        return csr;
    }
};