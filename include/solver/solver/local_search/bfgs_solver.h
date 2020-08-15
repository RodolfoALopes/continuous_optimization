#ifndef CONTINUOUS_OPTIMIZATION_BFGSSOLVER_H_
#define CONTINUOUS_OPTIMIZATION_BFGSSOLVER_H_

#include <iostream>
#include <Eigen/LU>
#include <solver/solver/solver_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/brent.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>

namespace solver {

    template<typename problem_type>
    class bfgs_solver : public solver_interface<problem_type, 1> {
    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::scalar;
        using typename superclass::vector_t;
        using typename superclass::hessian_t;

        void minimize(problem_type &objective_function, vector_t & x0) {
            const size_t DIM = x0.rows();
            hessian_t H = hessian_t::Identity(DIM, DIM);
            vector_t grad(DIM);
            vector_t x_old = x0;
            this->current_.reset();

            objective_function.gradient(x0, grad);
            this->current_.evaluations += DIM * 2;

            do {
                vector_t searchDir = -1 * H * grad;
                // check "positive definite"
                scalar phi = grad.dot(searchDir);

                // positive definit ?
                if ((phi > 0) || (phi != phi)) {
                    // no, we reset the hessian approximation
                    H = hessian_t::Identity(DIM, DIM);
                    searchDir = -1 * grad;
                }


                scalar rate;
                switch (this->options_.get_line_search()){
                    case line_search::AcceleratedStepSize : {
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, searchDir, objective_function,
                                this->max_step_size_line_search(objective_function, x0, searchDir),this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial());
                        break;
                    }
                    case line_search::Brent : {
                        rate = brent<problem_type, 1>::linesearch(x0, searchDir, objective_function,
                                this->max_step_size_line_search(objective_function, x0, searchDir), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::IntervalHalving : {
                        rate = interval_halving<problem_type, 1>::linesearch(x0, searchDir, objective_function,
                                this->max_step_size_line_search(objective_function, x0, searchDir), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::GoldenSection : {
                        rate = golden_section<problem_type, 1>::linesearch(x0, searchDir,
                                objective_function, this->max_step_size_line_search(objective_function, x0, searchDir), this->stop_, this->current_,
                                this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    default:
                        std::cerr << "Invalid line search algorithm." << std::endl;
                }


                x0 = x0 + rate * searchDir;

                scalar fx = objective_function.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

                vector_t grad_old = grad;
                objective_function.gradient(x0, grad);
                this->current_.evaluations += DIM * 2;
                vector_t s = rate * searchDir;
                vector_t y = grad - grad_old;

                const scalar rho = 1.0 / y.dot(s);
                H = H - rho * (s * (y.transpose() * H) + (H * y) * s.transpose()) + rho * (rho * y.dot(H * y) + 1.0)
                    * (s * s.transpose());

                if( (x_old-x0).template lpNorm<Eigen::Infinity>() < 1e-7  )
                    break;
                x_old = x0;
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