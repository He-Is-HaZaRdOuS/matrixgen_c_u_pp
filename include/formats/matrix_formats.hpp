#pragma once
#include <vector>

/* Lightweight CSR for I/O and GPU transfer */
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<double> values;
};