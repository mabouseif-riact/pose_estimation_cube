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



find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(COMPILE_FLAGS  ${COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(LINK_FLAGS ${LINK_FLAGS} ${MPI_LINK_FLAGS})


# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

# Link libraries
link_libraries(stdc++fs) # This is for filesystem header in order to read directories

# Add definitions
add_definitions(${PCL_DEFINITIONS})
add_definitions(${OpenCV_DEFINITIONS})


# Add executables
add_executable(generate_partial_views src/generate_partial_views.cpp src/visualization.cpp src/features.cpp src/registration.cpp src/helper_functions.cpp)
add_executable(train src/train.cpp src/visualization.cpp src/features.cpp src/registration.cpp src/helper_functions.cpp)
add_executable(test_query src/test_query.cpp src/visualization.cpp src/features.cpp src/registration.cpp src/helper_functions.cpp)

# Target include directories (https://stackoverflow.com/questions/4225005/how-to-generate-a-header-in-source-with-cmake)
target_include_directories(test_query PUBLIC
                                ${CMAKE_CURRENT_BINARY_DIR} ${MPI_LBRARIES}) # PROJECT_BINARY_DIRECTORY
target_include_directories(generate_partial_views PUBLIC
                                ${CMAKE_CURRENT_BINARY_DIR} ${MPI_LBRARIES}) # PROJECT_BINARY_DIRECTORY
target_include_directories(train PUBLIC
                                ${CMAKE_CURRENT_BINARY_DIR} ${MPI_LBRARIES}) # PROJECT_BINARY_DIRECTORY

# Target link libraries
target_link_libraries(generate_partial_views ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} mpi_cxx pthread)
target_link_libraries(train ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES} mpi_cxx pthread)
target_link_libraries(test_query ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES} mpi_cxx pthread)


