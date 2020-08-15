#include <iostream>
#include <solver/common.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/local_search/gradient_descent_solver.h>

using namespace solver;

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

    cout << "f in argmin: " << f(x0) << endl << endl;
    cout << "Solver status: " << solver.get_status() << endl << endl;
    cout << "Final criteria dfvalues: " << endl << solver.criteria() << endl << endl;
    cout << "x0: [";
    for(size_t i = 0; i < DIM-1; i++){
        cout << x0[i] << ", ";
    }
    cout << x0[DIM-1] << "]" << endl;

    return 0;
}