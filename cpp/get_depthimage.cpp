#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Global configuration
const std::string task = "um";
const std::string root_dir = "../";
const std::string out_dir = "./output";

// Read variable from calibration file
cv::Mat read_variable(const std::string& file_path, const std::string& name, int M, int N) {
    std::ifstream fid(file_path);
    std::string line;
    cv::Mat A = cv::Mat::zeros(M, N, CV_64F);

    if (!fid.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return A;
    }

    while (std::getline(fid, line)) {
        if (line.find(name + ":") == 0) { // Check if line starts with name:
            std::stringstream ss(line.substr(name.length() + 1)); // Skip prefix
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

// Load calibration parameters
// Returns: camera(P2 3x3), rotation(3x3), translation(3x1)
void load_calibration(const std::string& file, cv::Mat& camera, cv::Mat& rotation, cv::Mat& translation) {
    // Read P2 (we use P2 as part of camera intrinsic matrix)
    // In Python code: P2 = read_variable(..., 'P2', 3, 4) then take first 3 columns
    // Note: In KITTI, P0/P2 are 3x4 projection matrices (K * [R|t]).
    // Original Python logic: camera = P2[:, :3]
    cv::Mat P2_full = read_variable(file, "P2", 3, 4);
    camera = P2_full(cv::Rect(0, 0, 3, 3)).clone(); // Take first 3 columns (3x3)

    // Read Tr_velo_to_cam (3x4)
    cv::Mat B = read_variable(file, "Tr_velo_to_cam", 3, 4);
    
    // Original Python logic: R_rect = B[:3, :3]
    rotation = B(cv::Rect(0, 0, 3, 3)).clone();
    
    // Original Python logic: T_vc = B[:, 3]
    translation = B.col(3).clone();
}

// Read binary point cloud
cv::Mat load_pointcloud(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return cv::Mat();

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<float> buffer(size / sizeof(float));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        // N rows, 4 cols (x,y,z,i)
        int num_points = buffer.size() / 4;
        cv::Mat cloud(num_points, 4, CV_32F);
        memcpy(cloud.data, buffer.data(), size);
        return cloud;
    }
    return cv::Mat();
}

int main() {
    // Create output directory
    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    // Count files
    int num_files = 0;
    if (fs::exists(root_dir)) {
        for (const auto& entry : fs::directory_iterator(root_dir)) {
            std::string fname = entry.path().filename().string();
            if (fname.find(task + "_") == 0 && fname.find(".png") != std::string::npos) {
                num_files++;
            }
        }
    }
    std::cout << "Found " << num_files << " data file groups" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // Set distortion coefficients to 0

    for (int i = 0; i < num_files; ++i) {
        // Build file paths
        char buf[20];
        sprintf(buf, "%06d", i);
        std::string id_str = std::string(buf);

        std::string calib_path = root_dir + "/" + task + "_" + id_str + ".txt";
        std::string im_path = root_dir + "/" + task + "_" + id_str + ".png";
        std::string velo_path = root_dir + "/" + task + "_" + id_str + ".bin";
        std::string out_path = out_dir + "/" + task + "_" + id_str + "-depth.png";
        std::string stats_path = out_dir + "/" + task + "_" + id_str + "_depth_stats.txt";

        if (!fs::exists(calib_path) || !fs::exists(im_path) || !fs::exists(velo_path)) {
            std::cout << "Frame " << i << ": Missing files, skip." << std::endl;
            continue;
        }

        // 1. Load calibration
        cv::Mat camera_k, rot, trans;
        load_calibration(calib_path, camera_k, rot, trans);

        // 2. Load image and get dimensions
        cv::Mat image = cv::imread(im_path);
        if (image.empty()) continue;
        int h = image.rows;
        int w = image.cols;

        // 3. Load point cloud
        cv::Mat pointcloud = load_pointcloud(velo_path); // Nx4, CV_32F
        if (pointcloud.empty()) continue;

        // 4. Coordinate transformation: Lidar -> Camera
        // Python: transform = hstack(rotation, translation) -> 3x4
        //         point = dot(transform, point)
        // In C++, for efficiency, we build 4x4 matrix T
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        rot.copyTo(T(cv::Rect(0, 0, 3, 3)));
        trans.copyTo(T(cv::Rect(3, 0, 1, 3)));

        // Point cloud transform: P_cam = T * P_lidar
        // OpenCV matrix multiplication requires consistent types. pointcloud is 32F, T is 64F.
        pointcloud.convertTo(pointcloud, CV_64F); // Convert to double precision
        
        // Process homogeneous coordinates: point cloud is already Nx4 (x,y,z,1)
        // Python code: pointcloud[:, 3] = 1
        pointcloud.col(3) = 1.0;

        // Matrix multiplication: (4x4) * (4xN) -> (4xN)
        // Transpose pointcloud to 4xN
        cv::Mat pc_transposed = pointcloud.t();
        cv::Mat pc_cam = T * pc_transposed; // Result is 4xN
        
        // Transpose back to Nx4 for processing
        pc_cam = pc_cam.t(); // Nx4

        // 5. Filter points with z < 0
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
            std::cout << "Frame " << i << ": No valid points" << std::endl;
            continue;
        }

        // 6. Project 3D -> 2D
        // Python: cv2.projectPoints(..., rotation1, translation1, camera, distortion)
        // rotation1, translation1 are 0 because points are already in camera coordinate system
        std::vector<cv::Point2f> image_points;
        cv::Mat r_ident = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat t_ident = cv::Mat::zeros(3, 1, CV_64F);
        
        cv::projectPoints(valid_pts_3d, r_ident, t_ident, camera_k, distCoeffs, image_points);

        // 7. Filter points outside image and generate depth map
        cv::Mat depth_image = cv::Mat::zeros(h, w, CV_32F); // Initialize to 0
        
        // Statistics
        float min_d = 1e9, max_d = -1e9;
        double sum_d = 0;
        int valid_count = 0;

        for (size_t k = 0; k < image_points.size(); ++k) {
            int u = (int)image_points[k].x;
            int v = (int)image_points[k].y;
            float d = valid_depths[k];

            if (u >= 0 && u < w && v >= 0 && v < h) {
                depth_image.at<float>(v, u) = d;
                
                // Statistics
                if (d < min_d) min_d = d;
                if (d > max_d) max_d = d;
                sum_d += d;
                valid_count++;
            }
        }

        std::cout << "\nFrame " << std::setfill('0') << std::setw(6) << i << " Depth Info:" << std::endl;
        if (valid_count > 0) {
            std::cout << "  Min: " << min_d << "m, Max: " << max_d 
                      << "m, Avg: " << sum_d / valid_count << "m, Points: " << valid_count << std::endl;

            // Save statistics txt
            std::ofstream f_stats(stats_path);
            f_stats << "Depth Statistics - Frame " << i << "\n";
            f_stats << "Valid depth points: " << valid_count << "\n";
            f_stats << "Min depth: " << min_d << " meters\n";
            f_stats << "Max depth: " << max_d << " meters\n";
            f_stats << "Avg depth: " << sum_d / valid_count << " meters\n";
            f_stats.close();
        } else {
            std::cout << "  No valid depth points inside image" << std::endl;
        }

        // 8. Normalize and save
        cv::Mat depth_norm;
        if (valid_count > 0) {
            // Normalize values except 0.
            // Simple MinMax normalization treats 0 as min value, causing background not pure black.
            // Here to be consistent with Python code (cv2.NORM_MINMAX):
            cv::normalize(depth_image, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        } else {
            depth_norm = cv::Mat::zeros(h, w, CV_8U);
        }

        cv::imwrite(out_path, depth_norm);
        std::cout << "  Depth map saved" << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t2 - t1;
    std::cout << "\nAll frames processed!" << std::endl;
    std::cout << "Total processing time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}
