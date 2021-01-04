cmake_minimum_required(VERSION 3.5.1)
project(pygif LANGUAGES CXX)
find_package(OpenMP)

###################################################################
# LIBRARIES
###################################################################

# Include Eigen
include_directories(lib/eigen)

# Include nanoflann
include_directories(lib/nanoflann/include)

# Include pybind11
add_subdirectory(lib/pybind11)

###################################################################
# TARGET CONFIGURATION
###################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)
pybind11_add_module(pygif src/PythonBinding.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pygif PUBLIC OpenMP::OpenMP_CXX)
endif()