#pragma once
#include "matrix_formats.hpp"
#include <cuda.h>

/* GPU CSR format */
struct CSRDeviceMatrix {
    int rows, cols, nnz;
    int* row_ptr;
    int* col_ind;
    double* values;

    CSRDeviceMatrix() : rows(0), cols(0), nnz(0), row_ptr(nullptr), col_ind(nullptr), values(nullptr) {}
};

/* Prototype */
void allocate_device_matrix(const CSRMatrix& cpu_mat, CSRDeviceMatrix& device_mat);
void free_device_matrix(CSRDeviceMatrix& device_mat);
void lanczos_gpu(const CSRDeviceMatrix& device_input, CSRMatrix& output, double scale_x, double scale_y, int a, double threshold);