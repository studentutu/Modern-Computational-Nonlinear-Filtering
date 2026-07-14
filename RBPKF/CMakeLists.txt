cmake_minimum_required(VERSION 3.10)

# Create library target
add_library(rbpf_core
    src/resampling.cpp
)

# $<BUILD_INTERFACE>/$<INSTALL_INTERFACE> rather than bare source paths: rbpf_core
# is part of the exported nlf package, and install(EXPORT) rejects a target whose
# INTERFACE_INCLUDE_DIRECTORIES point into the build machine's source tree.
# The install root is the flat <prefix>/include/nlf, where install(DIRECTORY)
# places both these headers and Common/'s.
target_include_directories(rbpf_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/Common/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nlf>
)

target_compile_features(rbpf_core PUBLIC cxx_std_20)

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 3.4 REQUIRED)
endif()
target_link_libraries(rbpf_core PUBLIC Eigen3::Eigen)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(rbpf_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Enable LTO on library
enable_lto_if_supported(rbpf_core)

# Examples
add_executable(example_rbpf_ctrv examples/example_rbpf_ctrv.cpp)
target_link_libraries(example_rbpf_ctrv PRIVATE rbpf_core OptMathKernels)
enable_lto_if_supported(example_rbpf_ctrv)

# Tests
add_executable(test_rbpf_basic tests/test_rbpf_basic.cpp)
target_link_libraries(test_rbpf_basic PRIVATE rbpf_core OptMathKernels)
enable_lto_if_supported(test_rbpf_basic)
