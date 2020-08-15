#ifndef CONTINUOUS_OPTIMIZATION_CONJUGATED_GRADIENT_DESCENT_SOLVER_H_
#define CONTINUOUS_OPTIMIZATION_CONJUGATED_GRADIENT_DESCENT_SOLVER_H_

#include <Eigen/Core>
#include <solver/solver/solver_interface.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>
#include <solver/solver/line_search/brent.h>

namespace solver {

    template<typename problem_type>
    class conjugated_gradient_descent_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::scalar;
        using typename superclass::vector_t;

        void minimize(problem_type &objective_function, vector_t &x0) {
            vector_t grad(x0.rows());
            vector_t grad_old(x0.rows());
            vector_t Si(x0.rows());
            vector_t Si_old(x0.rows());

            this->current_.reset();
            do {

                objective_function.gradient(x0, grad);
                this->current_.evaluations += x0.rows() * 2;

                if (this->current_.iterations == 0) {
                    Si = -grad;
                } else {
                    double beta = grad.dot(grad) / (grad_old.dot(grad_old));
                    beta = (isnan(beta) ? 0 : beta);
                    Si = -grad + beta * Si_old;
                }

                scalar rate;

                switch (this->options_.get_line_search()){
                    case line_search::AcceleratedStepSize : {
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, Si, objective_function,
                                this->max_step_size_line_search(objective_function, x0, Si), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial());
                        break;
                    }
                    case line_search::Brent : {
                        rate = brent<problem_type, 1>::linesearch(x0, Si, objective_function, this->max_step_size_line_search(objective_function, x0, Si),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::IntervalHalving : {
                        rate = interval_halving<problem_type, 1>::linesearch(x0, Si, objective_function, this->max_step_size_line_search(objective_function, x0, Si),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::GoldenSection : {
                        rate = golden_section<problem_type, 1>::linesearch(x0, Si, objective_function, this->max_step_size_line_search(objective_function, x0, Si),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    default:
                        std::cerr << "Invalid line search algorithm." << std::endl;
                }

                vector_t x_copy = x0;

                x0 = x0 + rate * Si;

                scalar fx = objective_function.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

                grad_old = grad;
                Si_old = Si;

                this->current_.gradNorm = grad.template lpNorm<Eigen::Infinity>();

                ++this->current_.iterations;
                this->current_.fx_best = fx;
                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    std::cout << "Iterations: " << this->current_.iterations << " - Evaluations: " << this->current_.evaluations
                              << " - f(x) = " << this->current_.fx_best << std::endl;
                }
            } while (objective_function.callback(this->current_, x0) && (this->m_status == status::Continue));
        }
    };
}

#endif