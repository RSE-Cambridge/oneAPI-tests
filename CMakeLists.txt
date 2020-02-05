set(CMAKE_CXX_COMPILER "/opt/intel/inteloneapi/compiler/latest/linux/bin/dpcpp")

cmake_minimum_required(VERSION 3.0)
project(SODECL-oneAPI)

#set(MKLROOT "/opt/intel/inteloneapi/mkl/2021.1-beta03")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

include(dependencies/CMakeLists.txt)

add_executable(getPlatformsDevices source/getPlatformsDevices.cpp)
add_executable(example_1 source/example_1.cpp)
add_executable(euler_broomhead source/euler_broomhead.cpp)
add_executable(stoch_euler_broomhead source/stoch_euler_broomhead.cpp)

#target_compile_options(stoch_euler_broomhead PUBLIC -DMKL_ILP64)

target_link_libraries(getPlatformsDevices PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(example_1 PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(euler_broomhead PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(stoch_euler_broomhead PUBLIC OpenCL sycl cxxplot Chronos Python3::Python -L/opt/intel/inteloneapi/mkl/2021.1-beta03/lib/intel64 -lmkl_sequential -lmkl_core -lmkl_intel_ilp64 -lmkl_sycl)
