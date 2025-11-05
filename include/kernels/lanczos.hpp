#pragma once
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __CUDA_ARCH__
#define LANCZOS_INLINE __device__ __forceinline__
#else
#define LANCZOS_INLINE inline
#endif

LANCZOS_INLINE double lanczos_kernel(double x, int a = 3) {
    if (x == 0.0) return 1.0;
    if (fabs(x) > a) return 0.0;
    
    double pi_x = M_PI * x;
    double pi_x_over_a = pi_x / a;
    
    return a * sin(pi_x) * sin(pi_x_over_a) / (pi_x * pi_x);
}