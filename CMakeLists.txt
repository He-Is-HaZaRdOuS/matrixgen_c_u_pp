cmake_minimum_required(VERSION 3.18)
project(matrixgen VERSION 1.0 LANGUAGES CXX CUDA)

# Basic setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

set(COMMON_SOURCES
        src/cpu/lanczos.cpp
        src/gpu/lanczos.cu
        src/gpu/lanczos_sparse.cu
        src/gpu/lanczos_sparse2.cu
        src/gpu/lanczos_sparse3.cu
)

# Main application
add_executable(gpu_matgen src/main.cu ${COMMON_SOURCES})

# Test application
add_executable(test_real_matrices
        src/test_real_matrices.cu
        ${COMMON_SOURCES}
)

# Link libraries
foreach(target IN ITEMS gpu_matgen test_real_matrices)
    target_link_libraries(${target}
            OpenMP::OpenMP_CXX
            CUDA::cudart
    )

    target_include_directories(${target} PRIVATE
            include
    )

    target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --extended-lambda --expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endforeach()

# Resource path for tests
target_compile_definitions(test_real_matrices PRIVATE
        RESOURCES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/"
)

message(STATUS "Build configured successfully!")