#include <iostream>
#include <chrono>
#include <filesystem>
#include <string>
#include <sstream>

#include "formats/matrix_formats.hpp"
#include "formats/cuda_matrix_formats.cuh"
#include "utils/cuda_matrix_utils.cuh"
#include "io/matrix_market.hpp"
#include "io/matrix_visualizer.hpp"

/* Prototypes */
CSRMatrix lanczos_cpu_reference(const CSRMatrix& input,
                               double scale_x, double scale_y,
                               int a, double threshold);

CSRMatrix lanczos_sparse_omp(const CSRMatrix& input,
                             double scale_x, double scale_y,
                             int a, double threshold);

static std::string format_float_tag(double v)
{
    // snap very small values to zero
    if (std::fabs(v) < 1e-12)
        v = 0.0;

    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::fixed << std::setprecision(6) << v;

    std::string s = oss.str();

    // trim trailing zeros
    while (s.size() > 1 && s.back() == '0')
        s.pop_back();

    // trim trailing dot
    if (!s.empty() && s.back() == '.')
        s.pop_back();

    // make filename-safe: 2.5 -> 2p5
    for (char& c : s) {
        if (c == '.')
            c = 'p';
        else if (c == '-')
            c = 'm';
    }

    return s;
}


static std::string make_tagged_name(const std::string& base,
                                    const std::string& backend,
                                    double scale,
                                    int a)
{
    return base +
           "_" + backend +
           "_scale" + format_float_tag(scale) +
           "_a" + std::to_string(a);
}


void run_backend(const std::string& backend,
                 const std::string& matrix_path,
                 double scale,
                 int a,
                 double threshold)
{
    CSRMatrix input = MatrixMarketReader::load_csr(matrix_path);

    std::cout << "Input matrix\n";
    std::cout << " rows: " << input.rows
              << " cols: " << input.cols
              << " nnz: " << input.nnz
              << " density: "
              << (100.0 * input.nnz / (input.rows * input.cols))
              << "%\n";

    std::filesystem::create_directories(RESOURCES_PATH "plots");

    const std::string base =
        std::filesystem::path(matrix_path).stem().string();

    const std::string tag =
        make_tagged_name(base, backend, scale, a);

    std::filesystem::create_directories(RESOURCES_PATH "plots");
    std::filesystem::create_directories("results");

    /* Input visualization */
    MatrixVisualizer::save_spy_plot(
        input, RESOURCES_PATH "plots/" + base + "_input");

    MatrixVisualizer::save_heatmap(
        input, RESOURCES_PATH "plots/" + base + "_input_heatmap");

    /* Timing */
    auto t0 = std::chrono::high_resolution_clock::now();

    CSRMatrix output;

    if (backend == "cpu") {
        output = lanczos_cpu_reference(input, scale, scale, a, threshold);
    }
    else if (backend == "openmp") {
        output = lanczos_sparse_omp(input, scale, scale, a, threshold);
    }
    else if (backend == "cuda") {
        CSRDeviceMatrix d_input;
        allocate_device_matrix(input, d_input);

        lanczos_sparse_gpu_improved_chunked(
            d_input, output, scale, scale, a, threshold);

        free_device_matrix(d_input);
    }
    else {
        throw std::runtime_error("Unknown backend: " + backend);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    double density =
    100.0 * static_cast<double>(output.nnz) /
    (static_cast<double>(output.rows) *
     static_cast<double>(output.cols));

    std::cout << "\nExecution stats\n";
    std::cout << "  backend: " << backend << "\n";
    std::cout << "  scale: " << scale << "\n";
    std::cout << "  a: " << a << "\n";
    std::cout << "  time: " << ms << " ms\n";
    std::cout << "  rows: " << output.rows
              << " cols: " << output.cols
              << " nnz: " << output.nnz
              << " density: " << density << "%\n";

    /* Save matrix */
    const std::string out_path =
        "results/" + tag + ".mtx";

    MatrixMarketReader::save_csr(output, out_path);

    std::cout << "  saved: " << out_path << "\n";

    /* Visualizations */
    MatrixVisualizer::save_spy_plot(
        output,
        RESOURCES_PATH "plots/" + tag + "_spy");

    MatrixVisualizer::save_heatmap(
        output,
        RESOURCES_PATH "plots/" + tag + "_heatmap");
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr
            << "usage: " << argv[0]
            << " <cpu|openmp|cuda> <matrix_name> [scale] [a] [threshold]\n";
        return 1;
    }

    const std::string backend = argv[1];
    const std::string matrix_name = argv[2];

    const double scale =
        (argc > 3) ? std::stod(argv[3]) : 1.0;

    const int a =
        (argc > 4) ? std::stoi(argv[4]) : 2;

    const double threshold =
        (argc > 5) ? std::stod(argv[5]) : 1e-6;

    const std::string matrix_path =
        std::string(RESOURCES_PATH) + matrix_name;

    if (!std::filesystem::exists(matrix_path)) {
        std::cerr << "matrix not found: " << matrix_path << "\n";
        return 1;
    }

    run_backend(backend, matrix_path, scale, a, threshold);
    return 0;
}
