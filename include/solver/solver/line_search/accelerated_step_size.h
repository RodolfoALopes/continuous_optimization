#ifndef CONTINUOUS_OPTIMIZATION_ACCELERATED_STEP_SIZE_H
#define CONTINUOUS_OPTIMIZATION_ACCELERATED_STEP_SIZE_H

#include <solver/common.h>

namespace solver {

    template<typename problem_type, int Ord>
    class accelerated_step_size {

    public:
        using scalar = typename problem_type::scalar;
        using vector_t = typename problem_type::vector_t;

        static scalar linesearch(const vector_t &x, const vector_t &searchDir, problem_type &objective_function,
                                 scalar upper_bound, criteria<scalar> &stop_, criteria<scalar> &current_,
                                 status &m_status, stats<scalar> &m_stats, storage_criteria_level storage_criteria, const scalar alpha_init = 0.001) {
            scalar alpha = 0.0;
            scalar alpha_next = alpha_init;

            scalar fx_before = objective_function.value(x);
            ++current_.evaluations;
            m_stats.push(fx_before, current_.evaluations, storage_criteria);

            scalar fx_after;
            if(alpha_next <= upper_bound){
                fx_after = objective_function.value(x + alpha_next * searchDir);
                ++current_.evaluations;
                m_stats.push(fx_after, current_.evaluations, storage_criteria);
            }
            else{
                return 0.0;
            }

            m_status = checkConvergence(stop_, current_);
            while(fx_after <= fx_before && m_status == status::Continue){
                alpha = alpha_next;
                alpha_next = alpha_next * 2;
                fx_before = fx_after;

                if(alpha_next <= upper_bound) {
                    fx_after = objective_function.value(x + alpha_next * searchDir);
                    ++current_.evaluations;
                    m_stats.push(fx_after, current_.evaluations, storage_criteria);
                }
                else{
                    fx_after = max_limits<scalar>();
                }

                m_status = checkConvergence(stop_, current_);
            }

            return alpha;
        }

    };
}

#endif