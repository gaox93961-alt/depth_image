cmake_minimum_required(VERSION 3.10)
project(KittiDepthToolbox)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# 寻找 Eigen3 (只有 get_buquan 需要)
find_package(Eigen3 CONFIG REQUIRED)

# 2. 全局编译选项
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# 3. 定义第一个程序：生成深度图
add_executable(get_depthimage get_depthimage.cpp)
# 链接库：只需要 OpenCV
target_link_libraries(get_depthimage ${OpenCV_LIBS})

# 4. 定义第二个程序：补全深度图
add_executable(get_buquan get_buquan.cpp)
# 链接库：需要 OpenCV 和 Eigen3
target_link_libraries(get_buquan 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)