cmake_minimum_required(VERSION 3.10)
project(StereoVision)

set(CMAKE_CXX_STANDARD 17)

# Configuration OpenCV
set(OpenCV_DIR " C:/opencv/build") 
find_package(OpenCV REQUIRED)

# On indique à CMake où chercher les headers (.h)
include_directories(include)

# On liste les fichiers sources (.cpp)
set(SOURCE_FILES 
    src/main.cpp 
    src/calibration.cpp
    src/stereo_calibration.cpp
    src/feature_matcher.cpp
)

add_executable(MonApp ${SOURCE_FILES})
target_link_libraries(MonApp ${OpenCV_LIBS})