FILE(COPY ${CONTINUOUS_OPTIMIZATION}/external/cec2013_benchmark_functions/cec2013_cdatafiles DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY ${CONTINUOUS_OPTIMIZATION}/tests/benchmarks/experiment_01/initialsolutioncec2020 DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


# Experiments for LSGO - Black-Box Traditional Methods
add_executable(benchmark_lsgo_bob_traditional_methods benchmark_experiments_lsgo_bbo_traditional_methods.cpp)
target_link_libraries(benchmark_lsgo_bob_traditional_methods PUBLIC pthread thread_pool solver)

# Experiments for LSGO - Black-Box LSGSS
add_executable(benchmark_lsgo_bob_lsgss benchmark_experiments_lsgo_bbo_lsgss.cpp)
target_link_libraries(benchmark_lsgo_bob_lsgss PUBLIC pthread thread_pool solver)