# Gradient Descent to Solve the Sphere Problem Example
add_executable(example_sphere_I sphere_I.cpp)
target_link_libraries(example_sphere_I PUBLIC solver)

# Different Line Search combined with Gradient Descent to Solve the Sphere Problem Example
add_executable(example_sphere_II sphere_II.cpp)
target_link_libraries(example_sphere_II PUBLIC solver)

# All Local Search to Solve the Sphere Problem Example
add_executable(example_sphere_III sphere_III.cpp)
target_link_libraries(example_sphere_III PUBLIC solver)
