FILE(COPY ${CONTINUOUS_OPTIMIZATION}/external/cec2013_benchmark_functions/cec2013_cdatafiles DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
FILE(COPY ${CONTINUOUS_OPTIMIZATION}/tests/benchmarks/experiment_02/initialsolutions DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Experiments for LSGO - Black-Box LSGSS
add_executable(benchmark_bbo_lsgss benchmark_experiments_bbo_lsgss.cpp)
target_link_libraries(benchmark_bbo_lsgss PUBLIC solver)

# Experiments for LSGO - Black-Box Gradient Descent
add_executable(benchmark_bbo_gradient_descent benchmark_experiments_bbo_gradient_descent.cpp)
target_link_libraries(benchmark_bbo_gradient_descent PUBLIC pthread thread_pool solver)

# Experiments for LSGO - Black-Box Powell Method
add_executable(benchmark_bbo_powell benchmark_experiments_bbo_powell.cpp)
target_link_libraries(benchmark_bbo_powell PUBLIC solver)

# Experiments for LSGO - Gray-Box LSGSS
add_executable(benchmark_gbo_lsgss benchmark_experiments_gbo_lsgss.cpp)
target_link_libraries(benchmark_gbo_lsgss PUBLIC solver)

