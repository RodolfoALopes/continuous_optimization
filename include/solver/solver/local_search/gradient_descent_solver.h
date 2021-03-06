#ifndef CONTINUOUS_OPTIMIZATION_GRADIENT_DESCENT_SOLVER_H_
#define CONTINUOUS_OPTIMIZATION_GRADIENT_DESCENT_SOLVER_H_

#include <Eigen/Core>
#include <solver/solver/solver_interface.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>
#include <solver/solver/line_search/brent.h>

namespace solver {

    template <typename problem_type>
    class gradient_descent_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::scalar;
        using typename superclass::vector_t;

        void minimize(problem_type &objective_function, vector_t &x0) {

            vector_t direction(x0.rows());
            this->current_.reset();
            do {

                objective_function.gradient(x0, direction);
                this->current_.evaluations += x0.rows() * 2;

                scalar rate;

                direction = direction * -1;

                switch (this->options_.get_line_search()){
                    case line_search::AcceleratedStepSize : {
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, direction, objective_function,
                                this->max_step_size_line_search(objective_function, x0, direction), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial()) ;
                        break;
                    }
                    case line_search::Brent : {
                        rate = brent<problem_type, 1>::linesearch(x0, direction, objective_function, this->max_step_size_line_search(objective_function, x0, direction),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::IntervalHalving : {
                        rate = interval_halving<problem_type, 1>::linesearch(x0, direction, objective_function, this->max_step_size_line_search(objective_function, x0, direction),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::GoldenSection : {
                        rate = golden_section<problem_type, 1>::linesearch(x0, direction, objective_function, this->max_step_size_line_search(objective_function, x0, direction),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    default:
                        std::cerr << "Invalid line search algorithm." << std::endl;
                }

                x0 = x0 + rate * direction;

                scalar fx = objective_function.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

                this->current_.gradNorm = direction.template lpNorm<Eigen::Infinity>();
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