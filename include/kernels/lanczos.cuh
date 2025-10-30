#pragma once
#include <cuda_runtime.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Lanczos kernel function
__device__ inline double lanczos_kernel(double x, int a = 2) {
    if (x == 0.0) return 1.0;
    if (fabs(x) > a) return 0.0;
    return a * sin(M_PI * x) * sin(M_PI * x / a) / (M_PI * M_PI * x * x);
}

// Basic Lanczos kernel declaration
extern "C" {
    void lanczos_gpu_launcher(
        const int* row_ptr, const int* col_ind, const double* values,
        int input_rows, int input_cols,
        double* temp_dense, int output_rows, int output_cols,
        double scale_x, double scale_y, double threshold
    );
}