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
    vector_t x_copy = 100 * vector_t::Random(DIM);

    for(size_t i = 0; i < 4; i++) {
        gradient_descent_solver<sphere> solver;
        options<scalar> op = options<scalar>::defaults();
        switch(i) {
            case 0:{
                op.set_line_search(line_search::AcceleratedStepSize);
                cout << "Gradient Descent with Accelerated Step Size Line Search" << endl;
                break;
            }
            case 1:{
                cout << "Gradient Descent with Brent Line Search" << endl;
                op.set_line_search(line_search::Brent);
                break;
            }
            case 2:{
                cout << "Gradient Descent with Golden Section Line Search" << endl;
                op.set_line_search(line_search::GoldenSection);
                break;
            }
            case 3:{
                cout << "Gradient Descent with Interval Halving Line Search" << endl;
                op.set_line_search(line_search::IntervalHalving);
                break;
            }
            default:
                cerr << "Selected Line Search Algorithm invalid." << endl;
        }

        vector_t x0 = x_copy;

        solver.set_stop_criteria(criteria_);
        solver.set_options(op);
        solver.minimize(f, x0);

        cout << "f in argmin: " << f(x0) << endl;
        cout << "Solver status: " << solver.get_status() << endl << endl;
        cout << "Final criteria dfvalues: " << endl << solver.criteria();
        cout << "x0: [";
        for (size_t j = 0; j < DIM - 1; j++) {
            cout << x0[j] << ", ";
        }
        cout << x0[DIM - 1] << "]" << endl << endl << endl << endl;
    }
    return 0;
}