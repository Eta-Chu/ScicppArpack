set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -lstdc++ -pipe -O3 -fPIC")
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(ScicppArpack)

include_directories(~/Desktop/lib/xtensor/include)
include_directories(~/Desktop/lib/xtl/include)
include_directories(~/Desktop/lib/xtensor-blas/include)

file(GLOB_RECURSE SRC "./src/test.cpp")
add_executable(test ${SRC}) 
