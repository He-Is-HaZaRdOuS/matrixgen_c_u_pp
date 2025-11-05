#pragma once
#include <cuda.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) \
do { \
cudaError_t error = call; \
if (error != cudaSuccess) { \
std::stringstream ss; \
ss << "CUDA error at " << __FILE__ << ":" << __LINE__ \
<< " - " << cudaGetErrorString(error); \
throw std::runtime_error(ss.str()); \
} \
} while(0)

#define CUDA_CHECK_LAST_ERROR() \
do { \
cudaError_t error = cudaGetLastError(); \
if (error != cudaSuccess) { \
std::stringstream ss; \
ss << "CUDA kernel error at " << __FILE__ << ":" << __LINE__ \
<< " - " << cudaGetErrorString(error); \
throw std::runtime_error(ss.str()); \
} \
} while(0)