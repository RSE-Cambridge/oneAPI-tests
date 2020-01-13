set(CMAKE_CXX_COMPILER "/opt/intel/inteloneapi/compiler/latest/linux/bin/dpcpp")

cmake_minimum_required(VERSION 3.0)
project(SODECL-oneAPI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

include(dependencies/CMakeLists.txt)

add_executable(getPlatformsDevices source/getPlatformsDevices.cpp)
add_executable(example_1 source/example_1.cpp)
add_executable(euler source/euler.cpp)
add_executable(euler_implicit_data source/euler_implicit_data.cpp)

target_link_libraries(getPlatformsDevices OpenCL sycl)
target_link_libraries(example_1 OpenCL sycl)
target_link_libraries(euler OpenCL sycl)
target_link_libraries(euler_implicit_data OpenCL sycl)
