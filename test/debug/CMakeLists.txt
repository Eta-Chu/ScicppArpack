set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -lstdc++ -pipe -O3 -fPIC")
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(ScicppArpack)

include_directories(~/Desktop/lib/xtensor/include)
include_directories(~/Desktop/lib/xtl/include)
include_directories(~/Desktop/lib/xtensor-blas/include)
include_directories(/opt/homebrew/Cellar/arpack/3.9.1/libexec/include/arpack)
include_directories(../../inc)

file(GLOB_RECURSE SRC "./debug.cpp")
add_executable(test ${SRC}) 
target_link_libraries(test PUBLIC /opt/homebrew/Cellar/arpack/3.9.1/libexec/lib/libarpack.dylib)
