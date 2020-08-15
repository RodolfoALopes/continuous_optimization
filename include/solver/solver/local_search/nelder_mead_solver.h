#ifndef CONTINUOUS_OPTIMIZATION_NELDER_MEAD_SOLVER_H_
#define CONTINUOUS_OPTIMIZATION_NELDER_MEAD_SOLVER_H_

#include <cmath>
#include<iomanip>
#include <Eigen/Core>
#include <solver/solver/solver_interface.h>
#include <solver/common.h>

namespace solver {

    template<typename problem_type>
    class nelder_mead_solver : public solver_interface<problem_type, 0> {
    public:
        using superclass = solver_interface<problem_type, 0>;
        using typename superclass::scalar;
        using typename superclass::vector_t;
        using matrix_type = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;
        matrix_type x0;
        simplex_op lastOp = simplex_op::Place;
        bool initialSimplexCreated = false;


        template <class T = problem_type, class SCALAR = typename problem_type::scalar, int N = problem_type::Dim>
        std::enable_if_t<std::is_base_of_v<bounded_problem_interface<SCALAR,N>,T>,void>
        check_bound(const problem_type &problem, vector_t &x, const std::size_t dim) {
            for(size_t i = 0; i < dim; i++) {
                if(x[i] > problem.upperBound()[i]){
                    x[i] = problem.upperBound()[i];
                }
                else if(x[i] < problem.lowerBound()[i]){
                    x[i] = problem.lowerBound()[i];
                }
            }
            return;
        }

        template <class T = problem_type, class SCALAR = typename problem_type::scalar, int N = problem_type::Dim>
        std::enable_if_t<!std::is_base_of_v<bounded_problem_interface<SCALAR,N>,T>,void>
        check_bound(const problem_type &problem, vector_t &x, const std::size_t dim) {
            return;
        }

        matrix_type makeInitialSimplex(vector_t &x) {
            size_t DIM = x.rows();

            matrix_type s = matrix_type::Zero(DIM, DIM + 1);
            for (int c = 0; c < int(DIM) + 1; ++c) {
                for (int r = 0; r < int(DIM); ++r) {
                    s(r, c) = x(r);
                    if (r == c - 1) {
                        if (x(r) == 0) {
                            s(r, c) = 0.00025;
                        } else {
                            s(r, c) = (1 + 0.05) * x(r);
                        }
                    }
                }
            }

            return s;
        }

        void minimize(problem_type &objective_function, vector_t &x) {

            const scalar rho = 1.;    // rho > 0
            const scalar xi  = 2.;    // xi  > max(rho, 1)
            const scalar gam = 0.5;   // 0 < gam < 1

            const size_t DIM = x.rows();

            // create initial simplex
            if (! initialSimplexCreated) {
                x0 = makeInitialSimplex(x);
            }

            // compute function values
            std::vector<scalar> f; f.resize(DIM + 1);
            std::vector<int> index; index.resize(DIM + 1);
            for (int i = 0; i < int(DIM) + 1; ++i) {
                auto trial = static_cast<vector_t >(x0.col(i));
                check_bound(objective_function, trial, DIM);
                f[i] = objective_function(trial);
                x0.col(i) = trial;
                ++this->current_.evaluations;
                this->m_stats.push(f[i], this->current_.evaluations, this->options_.get_storage_criteria());
                index[i] = i;
            }

            sort(index.begin(), index.end(), [&](int a, int b)-> bool { return f[a] < f[b]; });

            this->m_status = checkConvergence(this->stop_, this->current_);

            while (objective_function.callback(this->current_, x0.col(index[0])) && (this->m_status == status::Continue)) {
                // conv-check
                scalar max1 = fabs(f[index[1]] - f[index[0]]);
                scalar max2 = (x0.col(index[1]) - x0.col(index[0]) ).array().abs().maxCoeff();
                for (int i = 2; i < int(DIM) + 1; ++i) {
                    scalar tmp1 = fabs(f[index[i]] - f[index[0]]);
                    if (tmp1 > max1)
                        max1 = tmp1;

                    scalar tmp2 = (x0.col(index[i]) - x0.col(index[0]) ).array().abs().maxCoeff();
                    if (tmp2 > max2)
                        max2 = tmp2;
                }
                const scalar tt1 = max(scalar(1.e-04), 10 * std::nextafter(f[index[0]], std::numeric_limits<double>::epsilon()) - f[index[0]]);
                const scalar tt2 = max(scalar(1.e-04), 10 * (std::nextafter(x0.col(index[0]).maxCoeff(), std::numeric_limits<double>::epsilon())
                                                                          - x0.col(index[0]).maxCoeff()));


                // midpoint of the simplex opposite the worst point
                vector_t x_bar = vector_t::Zero(DIM);
                for (int i = 0; i < int(DIM); ++i) {
                    x_bar += x0.col(index[i]);
                }
                x_bar /= scalar(DIM);

                // Compute the reflection point
                vector_t x_r  = ( 1. + rho ) * x_bar - rho   * x0.col(index[DIM]);
                check_bound(objective_function, x_r, DIM);
                const scalar f_r = objective_function(x_r);
                ++this->current_.evaluations;
                this->m_stats.push(f_r, this->current_.evaluations, this->options_.get_storage_criteria());

                lastOp = simplex_op::Reflect;

                if (f_r < f[index[0]]) {
                    // the expansion point
                    vector_t x_e = ( 1. + rho * xi ) * x_bar - rho * xi   * x0.col(index[DIM]);
                    check_bound(objective_function, x_e, DIM);
                    const scalar f_e = objective_function(x_e);
                    ++this->current_.evaluations;
                    this->m_stats.push(f_e, this->current_.evaluations, this->options_.get_storage_criteria());
                    if ( f_e < f_r ) {
                        // expand
                        lastOp = simplex_op::Expand;
                        x0.col(index[DIM]) = x_e;
                        f[index[DIM]] = f_e;
                    } else {
                        // reflect
                        lastOp = simplex_op::Reflect;
                        x0.col(index[DIM]) = x_r;
                        f[index[DIM]] = f_r;
                    }
                } else {
                    if ( f_r < f[index[DIM - 1]] ) {
                        x0.col(index[DIM]) = x_r;
                        f[index[DIM]] = f_r;
                    } else {
                        // contraction
                        if (f_r < f[index[DIM]]) {
                            vector_t x_c = (1 + rho * gam) * x_bar - rho * gam * x0.col(index[DIM]);
                            check_bound(objective_function, x_c, DIM);
                            const scalar f_c = objective_function(x_c);
                            ++this->current_.evaluations;
                            this->m_stats.push(f_c, this->current_.evaluations, this->options_.get_storage_criteria());
                            if ( f_c <= f_r ) {
                                // outside
                                x0.col(index[DIM]) = x_c;
                                f[index[DIM]] = f_c;
                                lastOp = simplex_op::ContractOut;
                            } else {
                                shrink(x0, index, f, objective_function);
                                lastOp = simplex_op::Shrink;
                            }
                        } else {
                            // inside
                            vector_t x_c = ( 1 - gam ) * x_bar + gam   * x0.col(index[DIM]);
                            check_bound(objective_function, x_c, DIM);
                            const scalar f_c = objective_function(x_c);
                            ++this->current_.evaluations;
                            this->m_stats.push(f_c, this->current_.evaluations, this->options_.get_storage_criteria());
                            if (f_c < f[index[DIM]]) {
                                x0.col(index[DIM]) = x_c;
                                f[index[DIM]] = f_c;
                                lastOp = simplex_op::ContractIn;
                            } else {
                                shrink(x0, index, f, objective_function);
                                lastOp = simplex_op::Shrink;
                            }
                        }
                    }
                }
                sort(index.begin(), index.end(), [&](int a, int b)-> bool { return f[a] < f[b]; });

                ++this->current_.iterations;
                this->current_.fx_best = f[index[0]];
                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    std::cout << "Iterations: " << this->current_.iterations << " - Evaluations: " << this->current_.evaluations
                              << " - f(x) = " << this->current_.fx_best << std::endl;
                }
            } // while loop

            // report the last result
            objective_function.detailed_callback(this->current_, lastOp, index[0], x0, f);
            x = x0.col(index[0]);
        }

        void shrink(matrix_type &x, std::vector<int> &index, std::vector<scalar> &f, problem_type &objective_function) {
            const scalar sig = 0.5;   // 0 < sig < 1
            const int DIM = x.rows();

            auto trial0 = static_cast<vector_t >(x.col(index[0]));
            check_bound(objective_function, trial0, DIM);
            f[index[0]] = objective_function(trial0);
            x.col(index[0]) = trial0;

            ++this->current_.evaluations;
            this->m_stats.push(f[index[0]], this->current_.evaluations, this->options_.get_storage_criteria());
            for (int i = 1; i < DIM + 1; ++i) {
                x.col(index[i]) = sig * x.col(index[i]) + (1. - sig) * x.col(index[0]);
                auto trial = static_cast<vector_t >(x.col(index[i]));
                check_bound(objective_function, trial, DIM);
                x.col(index[i]) = trial;
                f[index[i]] = objective_function(trial);
                ++this->current_.evaluations;
                this->m_stats.push(f[index[i]], this->current_.evaluations, this->options_.get_storage_criteria());
            }
        }
    };
}

#endif