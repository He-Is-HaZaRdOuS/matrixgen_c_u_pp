#include "formats/cuda_matrix_formats.cuh"
#include "utils/cuda_utils.cuh"

void allocate_device_matrix(const CSRMatrix& cpu_mat, CSRDeviceMatrix& device_mat) {
    device_mat.rows = cpu_mat.rows;
    device_mat.cols = cpu_mat.cols;
    device_mat.nnz = cpu_mat.nnz;

    CUDA_CHECK(cudaMalloc(&device_mat.row_ptr, (cpu_mat.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_mat.col_ind, cpu_mat.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_mat.values, cpu_mat.nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(device_mat.row_ptr, cpu_mat.row_ptr.data(),
                          (cpu_mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mat.col_ind, cpu_mat.col_ind.data(),
                          cpu_mat.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_mat.values, cpu_mat.values.data(),
                          cpu_mat.nnz * sizeof(double), cudaMemcpyHostToDevice));
}

void free_device_matrix(CSRDeviceMatrix& device_mat) {
    if (device_mat.row_ptr) CUDA_CHECK(cudaFree(device_mat.row_ptr));
    if (device_mat.col_ind) CUDA_CHECK(cudaFree(device_mat.col_ind));
    if (device_mat.values) CUDA_CHECK(cudaFree(device_mat.values));
    
    device_mat.row_ptr = nullptr;
    device_mat.col_ind = nullptr;
    device_mat.values = nullptr;
}