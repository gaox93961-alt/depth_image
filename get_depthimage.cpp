#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <windows.h>

namespace fs = std::filesystem;

// 全局配置
const std::string task = "um";
const std::string root_dir = "D:/depth-image-C";
const std::string out_dir = "D:/depth-image-C/output";

// 从标定文件读取变量
cv::Mat read_variable(const std::string& file_path, const std::string& name, int M, int N) {
    std::ifstream fid(file_path);
    std::string line;
    cv::Mat A = cv::Mat::zeros(M, N, CV_64F);

    if (!fid.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return A;
    }

    while (std::getline(fid, line)) {
        if (line.find(name + ":") == 0) { // 检查是否以 name: 开头
            std::stringstream ss(line.substr(name.length() + 1)); // 跳过前缀
            for (int i = 0; i < M * N; ++i) {
                double val;
                ss >> val;
                A.at<double>(i / N, i % N) = val;
            }
            break;
        }
    }
    return A;
}

// 加载标定参数
// 返回: camera(P2 3x3), rotation(3x3), translation(3x1)
void load_calibration(const std::string& file, cv::Mat& camera, cv::Mat& rotation, cv::Mat& translation) {
    // 读取 P0 (实际上我们代码用的是 P2 作为相机内参矩阵的一部分)
    // Python代码里: P2 = read_variable(..., 'P0', 3, 4) 然后取前3列
    // 注意：KITTI中 P0/P2 是 3x4 投影矩阵 (K * [R|t])。
    // 原Python逻辑: camera = P2[:, :3]
    cv::Mat P2_full = read_variable(file, "P0", 3, 4);
    camera = P2_full(cv::Rect(0, 0, 3, 3)).clone(); // 取前3列 (3x3)

    // 读取 Tr_velo_to_cam (3x4)
    cv::Mat B = read_variable(file, "Tr_velo_to_cam", 3, 4);
    
    // 原Python逻辑: R_rect = B[:3, :3]
    rotation = B(cv::Rect(0, 0, 3, 3)).clone();
    
    // 原Python逻辑: T_vc = B[:, 3]
    translation = B.col(3).clone();
}

// 读取二进制点云
cv::Mat load_pointcloud(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return cv::Mat();

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        // N行 4列 (x,y,z,i)
        int num_points = buffer.size() / 4;
        cv::Mat cloud(num_points, 4, CV_32F);
        memcpy(cloud.data, buffer.data(), size);
        return cloud;
    }
    return cv::Mat();
}

// 格式化文件名 (模拟 Python 的 %06d)
std::string format_path(const std::string& tmpl, int index) {
    char buffer[256];
    std::string format_str = tmpl + "/" + task + "_%06d";
    // 简单的根据后缀判断
    if (tmpl == root_dir) {
        // 需要根据上下文拼接后缀，这里简化处理，手动拼接完整路径
    }
    return ""; // 实际在主循环中处理更方便
}

int main() {
    SetConsoleOutputCP(65001);
    // 创建输出目录
    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    // 统计文件数量
    int num_files = 0;
    for (const auto& entry : fs::directory_iterator(root_dir)) {
        std::string fname = entry.path().filename().string();
        if (fname.find(task + "_") == 0 && fname.find(".png") != std::string::npos) {
            num_files++;
        }
    }
    std::cout << "找到 " << num_files << " 组数据文件" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // 畸变系数设为0

    for (int i = 0; i < num_files; ++i) {
        // 构建文件路径
        char buf[20];
        sprintf(buf, "%06d", i);
        std::string id_str = std::string(buf);

        std::string calib_path = root_dir + "/" + task + "_" + id_str + ".txt";
        std::string im_path = root_dir + "/" + task + "_" + id_str + ".png";
        std::string velo_path = root_dir + "/" + task + "_" + id_str + ".bin";
        std::string out_path = out_dir + "/" + task + "_" + id_str + ".png";
        std::string stats_path = out_dir + "/" + task + "_" + id_str + "_depth_stats.txt";

        if (!fs::exists(calib_path) || !fs::exists(im_path) || !fs::exists(velo_path)) {
            std::cout << "帧 " << i << ": 文件缺失，跳过。" << std::endl;
            continue;
        }

        // 1. 加载标定
        cv::Mat camera_k, rot, trans;
        load_calibration(calib_path, camera_k, rot, trans);

        // 2. 加载图像获取尺寸
        cv::Mat image = cv::imread(im_path);
        if (image.empty()) continue;
        int h = image.rows;
        int w = image.cols;

        // 3. 加载点云
        cv::Mat pointcloud = load_pointcloud(velo_path); // Nx4, CV_32F
        if (pointcloud.empty()) continue;

        // 4. 坐标变换: Lidar -> Camera
        // Python: transform = hstack(rotation, translation) -> 3x4
        //         point = dot(transform, point)
        // 在 C++ 中，为了高效，我们构建 4x4 矩阵 T
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        rot.copyTo(T(cv::Rect(0, 0, 3, 3)));
        trans.copyTo(T(cv::Rect(3, 0, 1, 3)));

        // 点云转换： P_cam = T * P_lidar
        // OpenCV 矩阵乘法需要类型一致。pointcloud是32F，T是64F。
        pointcloud.convertTo(pointcloud, CV_64F); // 转为双精度
        
        // 处理齐次坐标：点云已经是 Nx4 (x,y,z,1) (如果在Python里赋值了1)
        // Python代码: pointcloud[:, 3] = 1
        pointcloud.col(3) = 1.0;

        // 矩阵乘法: (4x4) * (4xN) -> (4xN)
        // 转置 pointcloud 变成 4xN
        cv::Mat pc_transposed = pointcloud.t();
        cv::Mat pc_cam = T * pc_transposed; // 结果是 4xN
        
        // 转回 Nx4 以便处理
        pc_cam = pc_cam.t(); // Nx4

        // 5. 过滤 z < 0 的点
        std::vector<cv::Point3f> valid_pts_3d;
        std::vector<float> valid_depths;
        
        for (int r = 0; r < pc_cam.rows; ++r) {
            double z = pc_cam.at<double>(r, 2);
            if (z >= 0) {
                valid_pts_3d.emplace_back(
                    (float)pc_cam.at<double>(r, 0),
                    (float)pc_cam.at<double>(r, 1),
                    (float)z
                );
                valid_depths.push_back((float)z);
            }
        }

        if (valid_pts_3d.empty()) {
            std::cout << "帧 " << i << ": 没有有效点" << std::endl;
            continue;
        }

        // 6. 投影 3D -> 2D
        // Python: cv2.projectPoints(..., rotation1, translation1, camera, distortion)
        // rotation1, translation1 是 0，因为点已经转换到相机坐标系了
        std::vector<cv::Point2f> image_points;
        cv::Mat r_ident = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat t_ident = cv::Mat::zeros(3, 1, CV_64F);
        
        cv::projectPoints(valid_pts_3d, r_ident, t_ident, camera_k, distCoeffs, image_points);

        // 7. 过滤图像范围外的点并生成深度图
        cv::Mat depth_image = cv::Mat::zeros(h, w, CV_32F); // 初始化为0
        
        // 统计信息
        float min_d = 1e9, max_d = -1e9;
        double sum_d = 0;
        int valid_count = 0;

        for (size_t k = 0; k < image_points.size(); ++k) {
            int u = (int)image_points[k].x;
            int v = (int)image_points[k].y;
            float d = valid_depths[k];

            if (u >= 0 && u < w && v >= 0 && v < h) {
                depth_image.at<float>(v, u) = d;
                
                // 统计
                if (d < min_d) min_d = d;
                if (d > max_d) max_d = d;
                sum_d += d;
                valid_count++;
            }
        }

        std::cout << "\n帧 " << std::setfill('0') << std::setw(6) << i << " 深度信息:" << std::endl;
        if (valid_count > 0) {
            std::cout << "  最小: " << min_d << "m, 最大: " << max_d 
                      << "m, 平均: " << sum_d / valid_count << "m, 点数: " << valid_count << std::endl;

            // 保存统计信息 txt
            std::ofstream f_stats(stats_path);
            f_stats << "深度统计信息 - 帧 " << i << "\n";
            f_stats << "有效深度点数: " << valid_count << "\n";
            f_stats << "最小深度: " << min_d << " 米\n";
            f_stats << "最大深度: " << max_d << " 米\n";
            f_stats << "平均深度: " << sum_d / valid_count << " 米\n";
            f_stats.close();
        } else {
            std::cout << "  没有落在图像内的有效深度点" << std::endl;
        }

        // 8. 归一化并保存
        cv::Mat depth_norm;
        if (valid_count > 0) {
            // Normalize 除了0以外的值。
            // 简单的 MinMax 归一化会将 0 视为最小值，导致背景不纯黑。
            // 这里为了保持和 Python 代码一致 (cv2.NORM_MINMAX):
            cv::normalize(depth_image, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        } else {
            depth_norm = cv::Mat::zeros(h, w, CV_8U);
        }

        cv::imwrite(out_path, depth_norm);
        std::cout << "  ✓ 深度图已保存" << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "\n✅ 所有帧处理完成!" << std::endl;
    std::cout << "总处理时间: " << elapsed.count() << " 秒" << std::endl;

    return 0;
}