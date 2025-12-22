#pragma once
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "formats/matrix_formats.hpp"

class MatrixVisualizer {
public:
    // Save spy plot as PPM image (no external dependencies!)
    static void save_spy_plot(const CSRMatrix& matrix, const std::string& filename) {
        std::cout << "Generating spy plot for " << matrix.rows << "x" << matrix.cols
                  << " matrix with " << matrix.nnz << " nonzeros..." << std::endl;

        int img_size = 1024;  // Output image size

        // Create white image
        std::vector<unsigned char> image(img_size * img_size * 3, 255);
        // std::cout << "Created image buffer: " << image.size() << " bytes" << std::endl;

        // Calculate scaling
        double scale_x = static_cast<double>(img_size) / matrix.cols;
        double scale_y = static_cast<double>(img_size) / matrix.rows;

        std::cout << "Visualization scale factors: " << scale_x << " x " << scale_y << std::endl;

        int pixels_drawn = 0;

        // Draw nonzeros as black pixels
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; j++) {
                int col = matrix.col_ind[j];

                int px = static_cast<int>(col * scale_x);
                int py = static_cast<int>(i * scale_y);

                if (px >= 0 && px < img_size && py >= 0 && py < img_size) {
                    int idx = (py * img_size + px) * 3;
                    image[idx] = 0;      // R
                    image[idx + 1] = 0;  // G
                    image[idx + 2] = 0;  // B
                    pixels_drawn++;
                }
            }
        }

        // std::cout << "Drew " << pixels_drawn << " pixels" << std::endl;

        // Write PPM file
        std::string ppm_file = filename + ".ppm";
        // std::cout << "Attempting to write to: " << ppm_file << std::endl;

        std::ofstream file(ppm_file, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << ppm_file << std::endl;
            return;
        }

        file << "P6\n" << img_size << " " << img_size << "\n255\n";
        file.write(reinterpret_cast<char*>(image.data()), image.size());

        if (file.fail()) {
            std::cerr << "ERROR: Failed to write image data" << std::endl;
        }

        file.close();

        // Check if file exists
        std::ifstream check(ppm_file);
        if (check.good()) {
            check.seekg(0, std::ios::end);
            std::size_t size = check.tellg();
            std::cout << "Spy plot saved to: " << ppm_file << " (" << size << " bytes)" << std::endl;
        } else {
            std::cerr << "ERROR: File was not created: " << ppm_file << std::endl;
        }
    }

    // Save heatmap with color mapping (for smaller matrices)
    static void save_heatmap(const CSRMatrix& matrix, const std::string& filename) {
        if (matrix.rows > 4096 || matrix.cols > 4096) {
            std::cout << "Matrix too large for heatmap (max 4096x4096)" << std::endl;
            return;
        }

        std::cout << "Generating heatmap for " << matrix.rows << "x" << matrix.cols
                  << " matrix..." << std::endl;

        int img_width = matrix.cols;
        int img_height = matrix.rows;

        // Find min/max values for color mapping
        double min_val = 0, max_val = 0;
        if (matrix.nnz > 0) {
            min_val = *std::min_element(matrix.values.begin(), matrix.values.end());
            max_val = *std::max_element(matrix.values.begin(), matrix.values.end());
        }

        double range = std::max(std::abs(min_val), std::abs(max_val));
        std::cout << "Value range: [" << min_val << ", " << max_val << "]" << std::endl;

        // Create image (white background)
        std::vector<unsigned char> image(img_width * img_height * 3, 255);

        // Map values to colors
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; j++) {
                int col = matrix.col_ind[j];
                double val = matrix.values[j];

                if (col >= 0 && col < img_width) {
                    int idx = (i * img_width + col) * 3;

                    // Blue-white-red colormap
                    if (val > 0) {
                        double t = val / range;
                        image[idx] = static_cast<unsigned char>(255);           // R
                        image[idx + 1] = static_cast<unsigned char>(255 * (1 - t)); // G
                        image[idx + 2] = static_cast<unsigned char>(255 * (1 - t)); // B
                    } else if (val < 0) {
                        double t = -val / range;
                        image[idx] = static_cast<unsigned char>(255 * (1 - t));     // R
                        image[idx + 1] = static_cast<unsigned char>(255 * (1 - t)); // G
                        image[idx + 2] = static_cast<unsigned char>(255);           // B
                    }
                    // else val == 0, stays white
                }
            }
        }

        // Write PPM
        std::string ppm_file = filename + ".ppm";
        // std::cout << "Writing heatmap to: " << ppm_file << std::endl;

        std::ofstream file(ppm_file, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << ppm_file << std::endl;
            return;
        }

        file << "P6\n" << img_width << " " << img_height << "\n255\n";
        file.write(reinterpret_cast<char*>(image.data()), image.size());
        file.close();

        std::ifstream check(ppm_file);
        if (check.good()) {
            check.seekg(0, std::ios::end);
            std::size_t size = check.tellg();
            std::cout << "Heatmap saved to: " << ppm_file << " (" << size << " bytes)" << std::endl;
        } else {
            std::cerr << "ERROR: File was not created: " << ppm_file << std::endl;
        }
    }
};