cmake_minimum_required(VERSION 3.9)
project(continuous_optimization)
set(CONTINUOUS_OPTIMIZATION ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(tests)