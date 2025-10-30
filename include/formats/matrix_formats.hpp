#pragma once
#include <vector>
#include <armadillo>

// CPU CSR format
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;

    // Convert from Armadillo sparse matrix
    static CSRMatrix from_arma(const arma::sp_mat& mat) {
        CSRMatrix csr;
        csr.rows = mat.n_rows;
        csr.cols = mat.n_cols;
        csr.nnz = mat.n_nonzero;

        csr.row_ptr.resize(csr.rows + 1);
        csr.col_ind.resize(csr.nnz);
        csr.values.resize(csr.nnz);

        // Build by scanning the matrix
        int count = 0;
        csr.row_ptr[0] = 0;

        for (int i = 0; i < csr.rows; i++) {
            int row_count = 0;
            for (int j = 0; j < csr.cols; j++) {
                double val = mat(i, j);
                if (val != 0.0) {
                    csr.col_ind[count] = j;
                    csr.values[count] = val;
                    count++;
                    row_count++;
                }
            }
            csr.row_ptr[i + 1] = csr.row_ptr[i] + row_count;
        }

        return csr;
    }

    arma::sp_mat to_arma() const {
        arma::sp_mat result(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                result(i, col_ind[j]) = values[j];
            }
        }

        return result;
    }
};

// Forward declaration for GPU matrix
struct CSRDeviceMatrix;

// Function to allocate GPU memory (implemented in CUDA files)
void allocate_device_matrix(const CSRMatrix& cpu_mat, CSRDeviceMatrix& device_mat);
void free_device_matrix(CSRDeviceMatrix& device_mat);