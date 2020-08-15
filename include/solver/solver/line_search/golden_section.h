#ifndef CONTINUOUS_OPTIMIZATION_GOLDEN_SECTION_H
#define CONTINUOUS_OPTIMIZATION_GOLDEN_SECTION_H

#include <solver/common.h>

namespace solver{

    template<typename problem_type, int Ord>
    class golden_section{

    public:
        using scalar = typename problem_type::scalar;
        using vector_t = typename problem_type::vector_t;

        static scalar linesearch(const vector_t &x, const vector_t &searchDir, problem_type &objective_function,
                                 scalar upper_bound, criteria<scalar> &stop_, criteria<scalar> &current_,
                                 status &m_status, stats<scalar> &m_stats, storage_criteria_level storage_criteria, const scalar tol = 1e-5) {

            scalar lower_bound = 0.0;

            scalar invphi = (std::sqrt(5.0) - 1.0) / 2.0;
            scalar invphi2 = (3 - std::sqrt(5.0)) / 2.0;

            scalar h = upper_bound - lower_bound;

            if(h <= tol) return 0.0;

            int n = int(std::ceil(std::log(tol / h) / std::log(invphi)));

            scalar c = lower_bound + invphi2 * h;
            scalar d = lower_bound + invphi * h;
            scalar fx_c = objective_function.value(x + c * searchDir);
            ++current_.evaluations;
            m_stats.push(fx_c, current_.evaluations, storage_criteria);
            scalar fx_d = objective_function.value(x + d * searchDir);
            ++current_.evaluations;
            m_stats.push(fx_d, current_.evaluations, storage_criteria);

            m_status = checkConvergence(stop_, current_);
            int i = 0;
            while(i < n-1 && m_status == status::Continue){
                if(fx_c < fx_d){
                    upper_bound = d;
                    d = c;
                    fx_d = fx_c;
                    h = invphi * h;
                    c = lower_bound + invphi2 * h;
                    fx_c = objective_function.value(x + c * searchDir);
                    ++current_.evaluations;
                    m_stats.push(fx_c, current_.evaluations, storage_criteria);
                }
                else{
                    lower_bound = c;
                    c = d;
                    fx_c = fx_d;
                    h = invphi * h;
                    d = lower_bound + invphi * h;
                    fx_d = objective_function.value(x + d * searchDir);
                    ++current_.evaluations;
                    m_stats.push(fx_d, current_.evaluations, storage_criteria);
                }
                i++;
                m_status = checkConvergence(stop_, current_);

                if(upper_bound < lower_bound){
                    scalar aux = lower_bound;
                    lower_bound = upper_bound;
                    upper_bound = aux;
                }
            }

            if (fx_c < fx_d){
                return (d + lower_bound) / 2;
            }
            else{
                return (upper_bound + c) / 2;
            }
        }
    };

}

#endif