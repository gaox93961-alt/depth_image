cmake_minimum_required(VERSION 3.10)
project(KittiDepthToolbox)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Linux编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 定义第一个程序：生成深度图
add_executable(get_depthimage get_depthimage.cpp)
target_link_libraries(get_depthimage ${OpenCV_LIBS})

# 定义第二个程序：补全深度图
add_executable(get_buquan get_buquan.cpp)
target_link_libraries(get_buquan 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
