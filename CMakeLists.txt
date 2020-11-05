# CMake version
cmake_minimum_required(VERSION 3.1)

# Project name and version
project(PoseEstimation VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

# Config
configure_file(src/config.h.in src/config.h)

# Find packages
find_package(PCL 1.7.2 REQUIRED)
find_package(OpenCV 3.2.0 REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})
add_definitions(${OpenCV_DEFINITIONS})


# Add executables
add_executable(main src/main.cpp src/visualization.cpp src/features.cpp src/registration.cpp src/helper_functions.cpp)

# Target include directories (https://stackoverflow.com/questions/4225005/how-to-generate-a-header-in-source-with-cmake)
target_include_directories(main PUBLIC
                                ${CMAKE_CURRENT_BINARY_DIR} ) # PROJECT_BINARY_DIRECTORY

# Target link libraries
target_link_libraries(main ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})


