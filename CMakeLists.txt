cmake_minimum_required(VERSION 3.17)

project(PathTracer LANGUAGES CXX CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 50)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/main.cu
)

# Add the executable
add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

# Add compile flags for C++ and CUDA
target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Werror>        # C++ flags
        $<$<COMPILE_LANGUAGE:CUDA>:> # CUDA flags
)

# Link CUDA libraries
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES})