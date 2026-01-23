#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <algorithm>

// OpenCV
#include <opencv2/opencv.hpp>

// Eigen (用于稀疏矩阵求解)
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

// Windows 终端编码
#define NOMINMAX
#include <windows.h>


namespace fs = std::filesystem;

// 定义路径 (请修改为你实际的路径)
const std::string base_dir = "D:/depth-image-C"; 
const std::string dir_rgb = base_dir + "/data_test/image";
const std::string dir_depth = base_dir + "/data_test/depth";
const std::string out_dir = base_dir + "/data_test/results";

// 类型别名
using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

// ---------------------------------------------------------
// 核心补全函数
// ---------------------------------------------------------
cv::Mat fill_depth_colorization(const cv::Mat& imgRgb, const cv::Mat& imgDepthInput, double alpha = 1.0) {
    // 1. 数据预处理
    cv::Mat imgDepth;
    double maxImgAbsDepth;
    cv::minMaxLoc(imgDepthInput, nullptr, &maxImgAbsDepth);

    if (maxImgAbsDepth == 0) return imgDepthInput.clone();

    // 归一化深度到 [0, 1]
    imgDepthInput.convertTo(imgDepth, CV_64F, 1.0 / maxImgAbsDepth);
    
    int H = imgDepth.rows;
    int W = imgDepth.cols;
    int numPix = H * W;

    // RGB 转 灰度 并归一化到 [0, 1] (重要：为了高斯权重计算正确)
    cv::Mat grayImg;
    if (imgRgb.channels() == 3) {
        cv::cvtColor(imgRgb, grayImg, cv::COLOR_BGR2GRAY);
    } else {
        grayImg = imgRgb.clone();
    }
    grayImg.convertTo(grayImg, CV_64F, 1.0 / 255.0);

    // 2. 准备构建稀疏矩阵
    std::vector<Triplet> coefficients;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(numPix);
    coefficients.reserve(numPix * 5); // 预估每个点有5个非零元素(自己+4邻居)

    // 3. 遍历像素构建方程
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            int idx = r * W + c;
            double depth_val = imgDepth.at<double>(r, c);

            // 如果是已知点 (Depth > 0)
            if (depth_val > 1e-6) {
                // 约束方程: alpha * x = alpha * depth
                coefficients.emplace_back(idx, idx, alpha);
                b(idx) = alpha * depth_val;
            } 
            else {
                // 如果是未知点 (Depth == 0)，利用邻域推导
                // 收集 3x3 邻域内的有效点 (排除中心自己)
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
                        // 权重计算 (和 Python 保持一致)
                        // exp( -(val1 - val2)^2 )
                        double diff = nGray - curGray;
                        double w = std::exp(-(diff * diff));
                        
                        weights.push_back(w);
                        neighbors.push_back(rr * W + cc);
                        w_sum += w;
                    }
                }

                // 如果没有邻居(孤立点?)，就设 x=0
                if (w_sum < 1e-9) {
                    coefficients.emplace_back(idx, idx, 1.0);
                    b(idx) = 0.0;
                } else {
                    // 方程: x_i - sum(w_j * x_j) = 0
                    // 对角线系数 1.0
                    coefficients.emplace_back(idx, idx, 1.0);
                    
                    // 邻居系数 -w_j / w_sum
                    for (size_t k = 0; k < neighbors.size(); ++k) {
                        coefficients.emplace_back(idx, neighbors[k], -weights[k] / w_sum);
                    }
                    // b(idx) 保持为 0
                }
            }
        }
    }

    // 4. 求解方程 Ax = b
    SpMat A(numPix, numPix);
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    // 使用 SparseLU 求解 (矩阵非对称)
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "矩阵分解失败!" << std::endl;
        return imgDepthInput.clone();
    }

    Eigen::VectorXd x = solver.solve(b);
    
    // 5. 还原结果
    cv::Mat result = cv::Mat::zeros(H, W, CV_64F);
    for (int i = 0; i < numPix; ++i) {
        int r = i / W;
        int c = i % W;
        result.at<double>(r, c) = x(i) * maxImgAbsDepth;
    }

    // 6. 融合：确保已知点保持原值 (可选，防止数值误差)
    // 实际上 solver 应该已经满足了软约束，但强制覆盖更稳妥
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            double raw = imgDepthInput.at<double>(r, c); // 注意 input 没归一化
            if (raw > 1e-6) {
                // 将原值写回，或者按比例融合。Python 代码是 output*(1-mask) + input
                result.at<double>(r, c) = raw; // 直接保留原值
            }
        }
    }

    return result;
}

// ---------------------------------------------------------
// 主函数
// ---------------------------------------------------------
int main() {
    SetConsoleOutputCP(65001); // 解决中文乱码

    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    // 获取所有需要处理的文件
    std::vector<std::string> filenames;
    for (const auto& entry : fs::directory_iterator(dir_rgb)) {
        std::string fname = entry.path().filename().string();
        // 简单过滤 png
        if (fname.length() > 4 && fname.substr(fname.length() - 4) == ".png") {
            filenames.push_back(fname);
        }
    }

    std::cout << "找到 " << filenames.size() << " 张图像，开始处理..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    // 串行处理 (如需并行，在 for 前加 #pragma omp parallel for)
    int count = 0;
    for (const auto& rgb_name : filenames) {
        // 构建路径
        std::string rgb_path = dir_rgb + "/" + rgb_name;
        
        // 构建深度图文件名: um_000000.png -> um_000000-depth.png
        std::string base_name = rgb_name.substr(0, rgb_name.length() - 4);
        std::string depth_name = base_name + "-depth.png";
        std::string depth_path = dir_depth + "/" + depth_name;

        if (!fs::exists(depth_path)) {
            // 尝试另一种命名可能? 
            // 如果文件名不匹配，请在这里调整逻辑
            std::cout << "跳过: 缺少深度图 " << depth_name << std::endl;
            continue;
        }

        std::cout << "[" << ++count << "/" << filenames.size() << "] 处理: " << rgb_name << " ... ";

        // 读取图像
        cv::Mat imgRgb = cv::imread(rgb_path);
        cv::Mat imgDepth = cv::imread(depth_path, cv::IMREAD_UNCHANGED); // 读取原始深度值

        if (imgRgb.empty() || imgDepth.empty()) {
            std::cout << "读取失败" << std::endl;
            continue;
        }

        // 确保深度图是 double 类型用于计算
        cv::Mat imgDepthDouble;
        imgDepth.convertTo(imgDepthDouble, CV_64F);

        // 执行补全
        cv::Mat resultDouble = fill_depth_colorization(imgRgb, imgDepthDouble);

        // 保存结果 (转为 uint8 或 uint16)
        // Python 代码最后保存为 uint8 (虽然深度图通常是 16位)
        // 这里为了和 Python 代码一致，归一化到 0-255 并存为 uint8
        // 如果你需要保存真实深度数据，建议存为 16U
        cv::Mat resultUint8;
        resultDouble.convertTo(resultUint8, CV_8U); 
        
        std::string save_path = out_dir + "/" + rgb_name;
        cv::imwrite(save_path, resultUint8);
        
        std::cout << "完成" << std::endl;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t_end - t_start;
    std::cout << "\n全部完成! 总耗时: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}