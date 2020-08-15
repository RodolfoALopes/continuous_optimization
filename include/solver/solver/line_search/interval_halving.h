#ifndef CONTINUOUS_OPTIMIZATION_INTERVAL_HALVING_H
#define CONTINUOUS_OPTIMIZATION_INTERVAL_HALVING_H

#include <iostream>
#include <solver/common.h>

namespace solver{

    template<typename problem_type, int Ord>
    class interval_halving{

    public:
        using scalar = typename problem_type::scalar;
        using vector_t = typename problem_type::vector_t;

        static scalar linesearch(const vector_t &x, const vector_t &searchDir, problem_type &objective_function,
                                 scalar upper_bound, criteria<scalar> &stop_, criteria<scalar> &current_,
                                 status &m_status, stats<scalar> &m_stats, storage_criteria_level storage_criteria, const scalar tol = 0.001) {
            scalar lower_bound = 0.0;

            scalar l = (upper_bound - lower_bound);
            scalar l_old;
            scalar alpha = 0.0;

            scalar step, step0, step1, step2;
            scalar fx0, fx1, fx2;

            step = l / 4;
            step0 = lower_bound + 2 * step;
            step1 = lower_bound + step;
            step2 = lower_bound + 3 * step;

            fx0 = objective_function.value(x + step0 * searchDir);
            ++current_.evaluations;
            m_stats.push(fx0, current_.evaluations, storage_criteria);

            do{
                l_old = l;

                fx1 = objective_function.value(x + step1 * searchDir);
                ++current_.evaluations;
                m_stats.push(fx1, current_.evaluations, storage_criteria);

                fx2 = objective_function.value(x + step2 * searchDir);
                ++current_.evaluations;
                m_stats.push(fx2, current_.evaluations, storage_criteria);

                if (fx2 > fx0 && fx0 > fx1) {
                    upper_bound = step0;
                    step0 = step1;
                    fx0 = fx1;
                } else if (fx2 < fx0 && fx0 < fx1) {
                    lower_bound = step0;
                    step0 = step2;
                    fx0 = fx2;
                } else if (fx1 > fx0 && fx2 > fx0) {
                    lower_bound = step1;
                    upper_bound = step2;
                } else{
                    return alpha;
                }

                if(upper_bound < lower_bound){
                    scalar aux = lower_bound;
                    lower_bound = upper_bound;
                    upper_bound = aux;
                }

                alpha = (upper_bound + lower_bound) / 2;

                l = (upper_bound - lower_bound);
                step = l / 4;
                step1 = lower_bound + step;
                step2 = lower_bound + 3 * step;

                m_status = checkConvergence(stop_, current_);

            }while(l_old != l && l > tol && m_status == status::Continue);

            return alpha;
        }
    };
}

#endif