#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <algorithm>

// OpenCV
#include <opencv2/opencv.hpp>

// Eigen (for sparse matrix solving)
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace fs = std::filesystem;

// Define paths (modify to your actual paths)
const std::string base_dir = "."; 
const std::string dir_rgb = base_dir + "/..";
const std::string dir_depth = base_dir + "/output";
const std::string out_dir = base_dir + "/results";

// Type aliases
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// ---------------------------------------------------------
// Core completion function
// ---------------------------------------------------------
cv::Mat fill_depth_colorization(const cv::Mat& imgRgb, const cv::Mat& imgDepthInput, double alpha = 1.0) {
    // 1. Data preprocessing
    cv::Mat imgDepth;
    double maxImgAbsDepth;
    cv::minMaxLoc(imgDepthInput, nullptr, &maxImgAbsDepth);

    if (maxImgAbsDepth == 0) return imgDepthInput.clone();

    // Normalize depth to [0, 1]
    imgDepthInput.convertTo(imgDepth, CV_64F, 1.0 / maxImgAbsDepth);
    
    int H = imgDepth.rows;
    int W = imgDepth.cols;
    int numPix = H * W;

    // RGB to Gray and normalize to [0, 1] (important for Gaussian weight calculation)
    cv::Mat grayImg;
    if (imgRgb.channels() == 3) {
        cv::cvtColor(imgRgb, grayImg, cv::COLOR_BGR2GRAY);
    } else {
        grayImg = imgRgb.clone();
    }
    grayImg.convertTo(grayImg, CV_64F, 1.0 / 255.0);

    // 2. Prepare to build sparse matrix
    std::vector<Triplet> coefficients;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(numPix);
    coefficients.reserve(numPix * 5); // Estimate 5 non-zero elements per point (self + 4 neighbors)

    // 3. Iterate pixels to build equations
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = r * W + c;
            double depth_val = imgDepth.at<double>(r, c);

            // If known point (Depth > 0)
            if (depth_val > 1e-6) {
                // Constraint equation: alpha * x = alpha * depth
                coefficients.emplace_back(idx, idx, alpha);
                b(idx) = alpha * depth_val;
            } 
            else {
                // If unknown point (Depth == 0), use neighborhood derivation
                // Collect valid points in 3x3 neighborhood (exclude center)
                int r_min = std::max(0, r - 1);
                int r_max = std::min(H - 1, r + 1);
                int c_min = std::max(0, c - 1);
                int c_max = std::min(W - 1, c + 1);

                double curGray = grayImg.at<double>(r, c);
                std::vector<int> neighbors;
                std::vector<double> weights;
                double w_sum = 0.0;

                for (int rr = r_min; rr <= r_max; ++rr) {
                    for (int cc = c_min; cc <= c_max; ++cc) {
                        if (rr == r && cc == c) continue;

                        double nGray = grayImg.at<double>(rr, cc);
                        // Weight calculation (consistent with Python)
                        // exp( -(val1 - val2)^2 )
                        double diff = nGray - curGray;
                        double w = std::exp(-(diff * diff));
                        
                        weights.push_back(w);
                        neighbors.push_back(rr * W + cc);
                        w_sum += w;
                    }
                }

                // If no neighbors (isolated point?), set x=0
                if (w_sum < 1e-9) {
                    coefficients.emplace_back(idx, idx, 1.0);
                    b(idx) = 0.0;
                } else {
                    // Equation: x_i - sum(w_j * x_j) = 0
                    // Diagonal coefficient 1.0
                    coefficients.emplace_back(idx, idx, 1.0);
                    
                    // Neighbor coefficient -w_j / w_sum
                    for (size_t k = 0; k < neighbors.size(); ++k) {
                        coefficients.emplace_back(idx, neighbors[k], -weights[k] / w_sum);
                    }
                    // b(idx) remains 0
                }
            }
        }
    }

    // 4. Solve equation Ax = b
    SpMat A(numPix, numPix);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    // Use SparseLU solver (matrix is non-symmetric)
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Matrix factorization failed!" << std::endl;
        return imgDepthInput.clone();
    }

    Eigen::VectorXd x = solver.solve(b);
    
    // 5. Restore result
    cv::Mat result = cv::Mat::zeros(H, W, CV_64F);
    for (int i = 0; i < numPix; ++i) {
        int r = i / W;
        int c = i % W;
        result.at<double>(r, c) = x(i) * maxImgAbsDepth;
    }

    // 6. Fusion: ensure known points keep original values (optional, prevent numerical errors)
    // Actually solver should have satisfied soft constraints, but forced override is safer
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            double raw = imgDepthInput.at<double>(r, c); // Note: input is not normalized
            if (raw > 1e-6) {
                // Write back original value, or blend proportionally. Python code is output*(1-mask) + input
                result.at<double>(r, c) = raw; // Directly keep original value
            }
        }
    }

    return result;
}

// ---------------------------------------------------------
// Main function
// ---------------------------------------------------------
int main() {
    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    // Get all files to process
    std::vector<std::string> filenames;
    if (fs::exists(dir_rgb)) {
        for (const auto& entry : fs::directory_iterator(dir_rgb)) {
            std::string fname = entry.path().filename().string();
            // Simple filter for png
            if (fname.length() > 4 && fname.substr(fname.length() - 4) == ".png") {
                filenames.push_back(fname);
            }
        }
    }

    std::cout << "Found " << filenames.size() << " images, start processing..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    // Serial processing (for parallel, add #pragma omp parallel for before for)
    int count = 0;
    for (const auto& rgb_name : filenames) {
        // Build paths
        std::string rgb_path = dir_rgb + "/" + rgb_name;
        
        // Build depth image filename: um_000000.png -> um_000000-depth.png
        std::string base_name = rgb_name.substr(0, rgb_name.length() - 4);
        std::string depth_name = base_name + "-depth.png";
        std::string depth_path = dir_depth + "/" + depth_name;

        if (!fs::exists(depth_path)) {
            // Try another naming possibility?
            // If filename doesn't match, adjust logic here
            std::cout << "Skip: missing depth map " << depth_name << std::endl;
            continue;
        }

        std::cout << "[" << ++count << "/" << filenames.size() << "] Processing: " << rgb_name << " ... ";

        // Read images
        cv::Mat imgRgb = cv::imread(rgb_path);
        cv::Mat imgDepth = cv::imread(depth_path, cv::IMREAD_UNCHANGED); // Read original depth values

        if (imgRgb.empty() || imgDepth.empty()) {
            std::cout << "Read failed" << std::endl;
            continue;
        }

        // Ensure depth map is double type for calculation
        cv::Mat imgDepthDouble;
        imgDepth.convertTo(imgDepthDouble, CV_64F);

        // Execute completion
        cv::Mat resultDouble = fill_depth_colorization(imgRgb, imgDepthDouble);

        // Save result (convert to uint8 or uint16)
        // Python code saves as uint8 (though depth maps are usually 16-bit)
        // Here to be consistent with Python code, normalize to 0-255 and save as uint8
        // If you need to save real depth data, recommend saving as 16U
        cv::Mat resultUint8;
        resultDouble.convertTo(resultUint8, CV_8U); 
        
        std::string save_path = out_dir + "/" + rgb_name;
        cv::imwrite(save_path, resultUint8);
        
        std::cout << "Done" << std::endl;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "\nAll done! Total time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
