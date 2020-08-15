#ifndef CONTINUOUS_OPTIMIZATION_LBFGSSOLVER_H_
#define CONTINUOUS_OPTIMIZATION_LBFGSSOLVER_H_

#include <iostream>
#include <Eigen/LU>
#include <solver/solver/solver_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/brent.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>


namespace solver {

    template<typename problem_type>
    class lbfgs_solver : public solver_interface<problem_type, 1> {
    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::scalar;
        using typename superclass::vector_t;
        using typename superclass::hessian_t;
        using matrix_type = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;

        void minimize(problem_type &objective_function, vector_t &x0) {
            this->current_.reset();

            const size_t m = 10;
            const size_t DIM = x0.rows();
            matrix_type sVector = matrix_type::Zero(DIM, m);
            matrix_type yVector = matrix_type::Zero(DIM, m);
            Eigen::Matrix<scalar, Eigen::Dynamic, 1> alpha = Eigen::Matrix<scalar, Eigen::Dynamic, 1>::Zero(m);
            vector_t grad(DIM), q(DIM), grad_old(DIM), s(DIM), y(DIM);
            objective_function.gradient(x0, grad);
            this->current_.evaluations += DIM * 2;
            vector_t x_old = x0;

            size_t iter = 0, globIter = 0;
            scalar H0k = 1;

            do {
                const scalar relativeEpsilon = static_cast<scalar>(0.0001) * std::max<scalar>(static_cast<scalar>(1.0), x0.norm());

                if (grad.norm() < relativeEpsilon)
                    break;

                //Algorithm 7.4 (L-BFGS two-loop recursion)
                q = grad;
                const int k = std::min<scalar>(m, iter);

                // for i = k − 1, k − 2, . . . , k − m§
                for (int i = k - 1; i >= 0; i--) {
                    // alpha_i <- rho_i*s_i^T*q
                    const double rho = 1.0 / static_cast<vector_t>(sVector.col(i))
                            .dot(static_cast<vector_t>(yVector.col(i)));
                    alpha(i) = rho * static_cast<vector_t>(sVector.col(i)).dot(q);
                    // q <- q - alpha_i*y_i
                    q = q - alpha(i) * yVector.col(i);
                }
                // r <- H_k^0*q
                q = H0k * q;
                //for i k − m, k − m + 1, . . . , k − 1
                for (int i = 0; i < k; i++) {
                    // beta <- rho_i * y_i^T * r
                    const scalar rho = 1.0 / static_cast<vector_t>(sVector.col(i))
                            .dot(static_cast<vector_t>(yVector.col(i)));
                    const scalar beta = rho * static_cast<vector_t>(yVector.col(i)).dot(q);
                    // r <- r + s_i * ( alpha_i - beta)
                    q = q + sVector.col(i) * (alpha(i) - beta);
                }
                // stop with result "H_k*f_f'=q"

                // any issues with the descent direction ?
                scalar descent = -grad.dot(q);
                scalar alpha_init =  1.0 / grad.norm();
                if (descent > -0.0001 * relativeEpsilon) {
                    q = -1 * grad;
                    iter = 0;
                    alpha_init = 1.0;
                }

                scalar rate;
                switch (this->options_.get_line_search()){
                    case line_search::AcceleratedStepSize : {
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, -q, objective_function,
                                this->max_step_size_line_search(objective_function, x0, -q), this->stop_, this->current_, this->m_status,
                                this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial()) ;
                        break;
                    }
                    case line_search::Brent : {
                        rate = brent<problem_type, 1>::linesearch(x0, -q, objective_function, this->max_step_size_line_search(objective_function, x0, -q),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::IntervalHalving : {
                        rate = interval_halving<problem_type, 1>::linesearch(x0, -q, objective_function, this->max_step_size_line_search(objective_function, x0, -q),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    case line_search::GoldenSection : {
                        rate = golden_section<problem_type, 1>::linesearch(x0, -q, objective_function, this->max_step_size_line_search(objective_function, x0, -q),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        break;
                    }
                    default:
                        std::cerr << "Invalid line search algorithm." << std::endl;
                }

                // update guess
                x0 = x0 - rate * q;

                scalar fx = objective_function.value(x0);
                ++this->current_.evaluations;
                this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

                grad_old = grad;
                objective_function.gradient(x0, grad);
                this->current_.evaluations += DIM * 2;

                s = x0 - x_old;
                y = grad - grad_old;

                // update the history
                if (iter < m) {
                    sVector.col(iter) = s;
                    yVector.col(iter) = y;
                } else {

                    sVector.leftCols(m - 1) = sVector.rightCols(m - 1).eval();
                    sVector.rightCols(1) = s;
                    yVector.leftCols(m - 1) = yVector.rightCols(m - 1).eval();
                    yVector.rightCols(1) = y;
                }
                // update the scaling factor
                H0k = y.dot(s) / static_cast<double>(y.dot(y));

                x_old = x0;

                iter++;
                globIter++;
                ++this->current_.iterations;
                this->current_.fx_best = fx;
                this->current_.gradNorm = grad.template lpNorm<Eigen::Infinity>();
                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    std::cout << "Iterations: " << this->current_.iterations << " - Evaluations: " << this->current_.evaluations
                              << " - f(x) = " << this->current_.fx_best << std::endl;
                }
            } while ((objective_function.callback(this->current_, x0)) && (this->m_status == status::Continue));
        }
    };
}

#endif