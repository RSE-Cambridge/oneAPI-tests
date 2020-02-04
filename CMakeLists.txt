set(CMAKE_CXX_COMPILER "/opt/intel/inteloneapi/compiler/latest/linux/bin/dpcpp")

cmake_minimum_required(VERSION 3.0)
project(SODECL-oneAPI)

set(MKLROOT "/opt/intel/inteloneapi/mkl/2021.1-beta03")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DMKL_ILP64")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${MKLROOT}/lib/intel64 -lmkl_sequential -lmkl_core")

include(dependencies/CMakeLists.txt)

add_executable(getPlatformsDevices source/getPlatformsDevices.cpp)
add_executable(example_1 source/example_1.cpp)
add_executable(euler source/euler.cpp)
add_executable(euler_broomhead source/euler_broomhead.cpp)
add_executable(stoch_euler_broomhead source/stoch_euler_broomhead.cpp)

#target_compile_options(stoch_euler_broomhead PUBLIC "-fsycl -DMKL_ILP64 -I${MKLROOT}/include")

target_link_libraries(getPlatformsDevices PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(example_1 PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(euler PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(euler_broomhead PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
target_link_libraries(stoch_euler_broomhead PUBLIC OpenCL sycl cxxplot Chronos Python3::Python)
