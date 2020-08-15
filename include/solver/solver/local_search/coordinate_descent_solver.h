#ifndef CONTINUOUS_OPTIMIZATION_COORDINATE_DESCENT_SOLVER_H
#define CONTINUOUS_OPTIMIZATION_COORDINATE_DESCENT_SOLVER_H

#include <iostream>
#include <solver/problem/problem_interface.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/solver_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>
#include <solver/solver/line_search/brent.h>

using namespace std;

namespace solver {

    template<typename problem_type>
    class coordinate_descent_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::vector_t;
        using typename superclass::scalar;
        using criteria_t = typename problem_type::criteria_t;

    public:

        coordinate_descent_solver(){}

        void minimize(problem_type &problem, vector_t &x0) {

            this->current_.reset();

            size_t dim = x0.size();
            scalar fx;

            fx = problem.value(x0);
            ++this->current_.evaluations;
            this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

            do{

                for(int i = 0; i < dim; i++){

                    vector_t grad = vector_t::Zero(dim);
                    problem.gradient(x0, grad, i);
                    grad = grad * -1;
                    this->current_.evaluations += 2;

                    scalar rate;
                    switch (this->options_.get_line_search()){
                        case line_search::AcceleratedStepSize : {
                            rate = accelerated_step_size<problem_type, 1>::linesearch(x0, grad, problem, this->max_step_size_line_search(problem, x0, grad),
                                    this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial());
                            break;
                        }
                        case line_search::Brent : {
                            rate = brent<problem_type, 1>::linesearch(x0, grad, problem, this->max_step_size_line_search(problem, x0, grad),
                                    this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                            break;
                        }
                        case line_search::IntervalHalving : {
                            rate = interval_halving<problem_type, 1>::linesearch(x0, grad, problem, this->max_step_size_line_search(problem, x0, grad),
                                    this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                            break;
                        }
                        case line_search::GoldenSection : {
                            rate = golden_section<problem_type, 1>::linesearch(x0, grad, problem, this->max_step_size_line_search(problem, x0, grad),
                                    this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                            break;
                        }
                        default:
                            std::cerr << "Invalid line search algorithm." << std::endl;
                    }
                    x0 = x0 + rate * grad;
                }
                fx = problem.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

                ++this->current_.iterations;
                this->current_.fx_best = fx;

                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    std::cout << "Iterations: " << this->current_.iterations << " - Evaluations: " << this->current_.evaluations
                              << " - f(x) = " << this->current_.fx_best << std::endl;
                }

            }while(problem.callback(this->current_, x0) && (this->m_status == status::Continue));
        }
    };
}
#endif