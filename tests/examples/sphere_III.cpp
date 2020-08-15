#include <iostream>
#include <solver/common.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/local_search/bfgs_solver.h>
#include <solver/solver/local_search/conjugated_gradient_descent_solver.h>
#include <solver/solver/local_search/coordinate_descent_solver.h>
#include <solver/solver/local_search/gradient_descent_solver.h>
#include <solver/solver/local_search/lbfgs_solver.h>
#include <solver/solver/local_search/lsgss_solver.h>
#include <solver/solver/local_search/mts1_solver.h>
#include <solver/solver/local_search/nelder_mead_solver.h>
#include <solver/solver/local_search/newton_descent_solver.h>
#include <solver/solver/local_search/powell_solver.h>

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
    cout << "f(x0): " << f.value(x_copy) << endl;
    cout << "x0: [";
    for (size_t j = 0; j < DIM - 1; j++) {
        cout << x_copy[j] << ", ";
    }
    cout << x_copy[DIM - 1] << "]" << endl << endl << endl << endl;

    for(size_t i = 0; i < 10; i++) {
        vector_t x0 = x_copy;
        options<scalar> op = options<scalar>::defaults();
        op.set_line_search(line_search::IntervalHalving);

        switch(i) {
            case 0:{
                cout << "BFGS - Algorithm: " << endl;
                bfgs_solver<sphere> solver;
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
                break;
            }
            case 1:{
                cout << "Conjugated Gradient Descent - Algorithm: " << endl;
                conjugated_gradient_descent_solver<sphere> solver;
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
                break;
            }
            case 2:{
                cout << "Coordinate Descent - Algorithm: " << endl;
                coordinate_descent_solver<sphere> solver;
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
                break;
            }
            case 3:{
                cout << "Gradient Descent - Algorithm: " << endl;
                gradient_descent_solver<sphere> solver;
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
                break;
            }
            case 4:{
                cout << "Limited Memory BFGS - Algorithm: " << endl;
                lbfgs_solver<sphere> solver;
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
                break;
            }
            case 5:{
                cout << "LSGSS - Algorithm: " << endl;
                lsgss_solver<sphere> solver;
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
                break;
            }
            case 6:{
                cout << "MTS - Algorithm: " << endl;
                mts1_solver<sphere> solver;
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
                break;
            }
            case 7:{
                cout << "Nelder-Mead - Algorithm: " << endl;
                nelder_mead_solver<sphere> solver;
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
                break;
            }
            case 8:{
                cout << "Newton Descent - Algorithm: " << endl;
                newton_descent_solver<sphere> solver;
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
                break;
            }
            case 9:{
                cout << "Powell's Algorithm: " << endl;
                powell_solver<sphere> solver;
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
                break;
            }
            default:
                cerr << "Selected Local Search Invalid." << endl;
        }
    }
    return 0;
}