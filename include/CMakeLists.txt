add_library(solver INTERFACE)
target_include_directories(solver INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(solver INTERFACE eigen cec_benchmark)