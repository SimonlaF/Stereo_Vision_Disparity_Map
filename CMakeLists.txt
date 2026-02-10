cmake_minimum_required(VERSION 3.10)
project(StereoVision)

set(CMAKE_CXX_STANDARD 17)

# Pointer vers le dossier qui contient OpenCVConfig.cmake pour Visual Studio
# C'est souvent dans build/x64/vc16/lib ou build
set(OpenCV_DIR "C:/opencv/build") 

find_package(OpenCV REQUIRED)

add_executable(MonApp src/main.cpp)
target_link_libraries(MonApp ${OpenCV_LIBS})