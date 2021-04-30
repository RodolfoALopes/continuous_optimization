# Continuous Optimization Library

## Description

This continuous optimization project is a C++ library based on CppOptimizationLibrary [1]. 

* [Requirements](#requirements)
* [Algorithms](#algorithms)
    * [Line Search Methods](#line-search-methods)
    * [Local Search Methods](#local-search-methods)
    * [Global Search Methods](#global-search-methods)
    * [Decomposition Methods](#decomposition-methods)
* [Example](#example)
* [Benchmark Functions](#benchmark-functions)
* [Thanks](#thanks)
* [References](#references)

## Requirements

C++ 17

Cmake - Minimun Version 3.9

## Algorithms

## Black-Box Approaches

#### Line Search Methods
- Accelerated Step Size (ASS) [4]
- Brent's Method (BM) [5]
- Golden Section (GS) [4]
- Interval Halving (IH) [4]


#### Local Search Methods
- Broyden - Fletcher - Goldfarb - Shanno (BFGS) [1]
- Conjugate Gradient (CG) [1]
- Coordinate Descent (CD) [3]
- Gradient Descent (GD) [1]
- Local Search with Groups of Step Sizes (LSGSS) [10]
- Limited Memory Broyden - Fletcher - Goldfarb - Shanno (L-BFGS) [1, 6]
- Multiple Trajectory Search (MTS1) [7, 8]
- Nelder-Mead (NM) [1]
- Newton Descent (ND) [1]
- Powell’s Method (PM) [4]

#### Global Search Methods
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) (*not available yet*)
- Differential Evolution (DE) (*not available yet*)

#### Decomposition Methods
 - Differential Grouping (DG) (*not available yet*)
 - Extended Differential Grouping (XDG) (*not available yet*)
 - Fast Interdependency Identification (FII) (*not available yet*)
 - Global Differential Grouping (GDG) (*not available yet*)
 - Recursive Differential Grouping 3 (RDG-3) (*not available yet*)

## Gray-Box Approaches

#### Local Search Methods
- Gray-Box Version of Local Search with Groups of Step Sizes (GBO-LSGSS)
 
## Example

For very quick complete examples, see the directory [examples](tests/examples/).

### Implementing an Optimization Problem

The code below presents an of the Sphere Problem [9]:

```cpp
    template<typename T>
    class sphere : public bounded_problem_interface<T> {
    public:
        using Superclass = bounded_problem_interface<T>;
        using typename Superclass::vector_t;
    
    public:
        explicit sphere(int dim) : Superclass(dim) {
            this->setLowerBound(vector_t::Ones(dim) * -100.0);
            this->setUpperBound(vector_t::Ones(dim) * 100.0);
        }
    
        T value(const vector_t &x) override {
            T sum = 0.0;
            for(int i = 0; i < x.size(); i++){
                sum += x[i] * x[i];
            }
            return sum;
        }
    };
```

### Solving an Optimization Problem

The code below presents a simple example of the Gradient Descent method to solve the Sphere Problem previously described.

```cpp
    int main(){
        solver::criteria<double> criteria_ = solver::criteria<double>::defaults();
        criteria_.iterations = 1000;
        criteria_.evaluations = 100000;
        criteria_.gradNorm = -0.1;
        criteria_.fx_is_know = true;
        criteria_.fx_best = 0.0;
    
        const size_t DIM = 10;
        typedef double scalar;
        typedef sphere<scalar> sphere;
        typedef typename sphere::vector_t vector_t;
    
        sphere f(DIM);
        vector_t x0 = 100 * vector_t::Random(DIM);
    
        gradient_descent_solver<sphere> solver;
        options<scalar> op = options<scalar>::defaults();
        solver.set_stop_criteria(criteria_);
        solver.set_options(op);
    
        solver.minimize(f, x0);
    
        std::cout << "f in argmin: " << f(x0) << std::endl << endl;
        std::cout << "Solver status: " << solver.get_status() << std::endl << std::endl;
        std::cout << "Final criteria dfvalues: " << std::endl << solver.criteria() << std::endl << std::endl;
        std::cout << "x0: [";
        for(size_t i = 0; i < DIM-1; i++){
            std::cout << x0[i] << ", ";
        }
        std::cout << x0[DIM-1] << "]" << std::endl;
    
        return 0;
    }
```

Note that the continuous optimization algorithm can be easily changed just replacing the C++ class of the solver, for example, replacing *gradient_descent_solver* to *lsgss_solver*.

## Benchmark Functions
* Large Scale Global Optimization (LSGO) CEC'2013 Benchmark [2]


## Thanks
We would like to thank the developers of these libraries:

- [CppOptimizationLibrary](https://github.com/PatWie/CppNumericalSolvers)

## References
[1] P. Wieschollek. [*CppOptimizationLibrary*](https://github.com/PatWie/CppNumericalSolvers). (2016)

[2] X. Li, K. Tang, M. N. Omidvar, Z. Yang, K. Qin. *Benchmark functions for the cec’2013 special session and competition on large-scale global optimization*. (2013)

[3] S. J. Wright. *Coordinate Descent Algorithms.* Mathematical Programming. (2015)

[4] S. Rao. *Engineering Optimization: Theory and Practice: Fourth Edition.* John Wiley and Sons (2009)

[5] Boost C++ Library. [*Locating Function Minima using Brent's algorithm*](https://www.boost.org/doc/libs/1_65_0/libs/math/doc/html/math_toolkit/roots/brent_minima.html) Access in 08-11-2020. (2020)

[6] J. Nocedal and S. J. Wright. *L-BFGS: Numerical Optimization, 2nd ed.* New York: Springer. (2006)

[7] L. Y. Tseng and C. Chen. *Multiple Trajectory Search for Large Scale Global Optimization.* IEEE Congress on Evolutionary Computation (CEC). (2008)

[8] D. Molina, A. LaTorre and F. Herrera, *SHADE with Iterative Local Search for Large-Scale Global Optimization.* IEEE Congress on Evolutionary Computation (CEC). (2018)

[9] Virtual Library of Simulation Experiments. *Test Functions and Datasets - Sphere Function*. Access 08-11-2020. (2020)

[10] R. A. Lopes, A. R. R. Freitas, R. C. P. Silva. *Local Search with Groups of Step Sizes* Operations Research Letters. (2021) 
