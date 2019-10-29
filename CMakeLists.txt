cmake_minimum_required(VERSION 3.10)

project(LineRenderer)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(line_renderer src/main.cpp src/wu_line.hpp)
target_link_libraries(line_renderer ${OpenCV_LIBS})