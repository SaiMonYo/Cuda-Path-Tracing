cmake_minimum_required(VERSION 3.17)

project(PathTracer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 50)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cu
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES})
#set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)