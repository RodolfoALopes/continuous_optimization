#ifndef CONTINUOUS_OPTIMIZATION_NEWTON_DESCENT_SOLVER_H_
#define CONTINUOUS_OPTIMIZATION_NEWTON_DESCENT_SOLVER_H_

#include <iostream>
#include <Eigen/LU>
#include <solver/solver/solver_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/brent.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>

namespace solver {

    template<typename problem_type>
    class newton_descent_solver : public solver_interface<problem_type, 2> {
    public:
        using superclass = solver_interface<problem_type, 2>;
        using typename superclass::scalar;
        using typename superclass::vector_t;
        using typename superclass::hessian_t;

        void minimize(problem_type &objective_function, vector_t &x0) {
            const int DIM = x0.rows();
            vector_t grad = vector_t::Zero(DIM);
            hessian_t hessian = hessian_t::Zero(DIM, DIM);
            this->current_.reset();
            do {
                objective_function.gradient(x0, grad);
                this->current_.evaluations += DIM * 2;

                objective_function.hessian(x0, hessian);
                this->current_.evaluations += this->current_.evaluations + ((DIM * 2) * (DIM * 2));

                hessian += (1e-5) * hessian_t::Identity(DIM, DIM);
                vector_t delta_x = hessian.lu().solve(-grad);

                scalar rate;
                switch (this->options_.get_line_search()){
                    case line_search::AcceleratedStepSize : {
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, delta_x, objective_function,
                                this->max_step_size_line_search(objective_function, x0, delta_x), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial()) ;
                        break;
                    }
                    case line_search::Brent : {
                        rate = brent<problem_type, 1>::linesearch(x0, delta_x, objective_function,
                                this->max_step_size_line_search(objective_function, x0, delta_x),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::IntervalHalving : {
                        rate = interval_halving<problem_type, 1>::linesearch(x0, delta_x, objective_function,
                                this->max_step_size_line_search(objective_function, x0, delta_x),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::GoldenSection : {
                        rate = golden_section<problem_type, 1>::linesearch(x0, delta_x, objective_function,
                                this->max_step_size_line_search(objective_function, x0, delta_x),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    default:
                        std::cerr << "Invalid line search algorithm." << std::endl;
                }

                x0 = x0 + rate * delta_x;
                scalar fx = objective_function.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());


                ++this->current_.iterations;
                this->current_.fx_best = fx;
                this->current_.gradNorm = grad.template lpNorm<Eigen::Infinity>();
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