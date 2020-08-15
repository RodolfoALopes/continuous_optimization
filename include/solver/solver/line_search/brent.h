#ifndef CONTINUOUS_OPTIMIZATION_BRENT_H
#define CONTINUOUS_OPTIMIZATION_BRENT_H

#include <solver/common.h>

namespace solver{

    template<typename problem_type, int Ord>
    class brent{

    public:

        using scalar = typename problem_type::scalar;
        using vector_t = typename problem_type::vector_t;

        static scalar linesearch(const vector_t &x0, const vector_t &searchDir, problem_type &objective_function,
                                 scalar upper_bound,  criteria<scalar> &stop_, criteria<scalar> &current_,
                                 status &m_status, stats<scalar> &m_stats, storage_criteria_level storage_criteria,
                                 const scalar tol = 0.001, const size_t count_max = 100){

            scalar lower_bound = 0.0;

            scalar x;
            scalar w;
            scalar v;
            scalar u;
            scalar delta;
            scalar delta2;
            scalar fu, fv, fw, fx;
            scalar mid;
            scalar fract1, fract2;

            static const scalar golden = 0.3819660f;

            x = w = v = upper_bound;
            fw = fv = fx = objective_function.value(x0 + x * searchDir);
            ++current_.evaluations;
            m_stats.push(fx, current_.evaluations, storage_criteria);

            delta2 = delta = 0;

            size_t count = count_max;

            do{
                mid = (lower_bound + upper_bound) / 2;
                fract1 = tol * fabs(x) + tol / 4;
                fract2 = 2 * fract1;
                if(fabs(x - mid) <= (fract2 - (upper_bound - lower_bound) / 2))
                    break;

                if(fabs(delta2) > fract1){

                    scalar r = (x - w) * (fx - fv);
                    scalar q = (x - v) * (fx - fw);
                    scalar p = (x - v) * q - (x - w) * r;
                    q = 2 * (q - r);
                    if(q > 0)
                        p = -p;
                    q = fabs(q);
                    scalar td = delta2;
                    delta2 = delta;
                    if((fabs(p) >= fabs(q * td / 2)) || (p <= q * (lower_bound - x)) || (p >= q * (upper_bound - x))) {
                        delta2 = (x >= mid) ? lower_bound - x : upper_bound - x;
                        delta = golden * delta2;
                    }
                    else{
                        delta = p / q;
                        u = x + delta;
                        if(((u - lower_bound) < fract2) || ((upper_bound- u) < fract2))
                            delta = (mid - x) < 0 ? (scalar)-fabs(fract1) : (scalar)fabs(fract1);
                    }
                }
                else{
                    delta2 = (x >= mid) ? lower_bound - x : upper_bound - x;
                    delta = golden * delta2;
                }

                u = (fabs(delta) >= fract1) ? scalar(x + delta) : (delta > 0 ? scalar(x + fabs(fract1)) : scalar(x - fabs(fract1)));
                fu = objective_function.value(x0 + u * searchDir);
                ++current_.evaluations;
                m_stats.push(fu, current_.evaluations, storage_criteria);

                if(fu <= fx){
                    if(u >= x)
                        lower_bound = x;
                    else
                        upper_bound = x;
                    v = w;
                    w = x;
                    x = u;
                    fv = fw;
                    fw = fx;
                    fx = fu;
                }
                else{
                    if(u < x)
                        lower_bound = u;
                    else
                        upper_bound = u;
                    if((fu <= fw) || (w == x)){
                        v = w;
                        w = u;
                        fv = fw;
                        fw = fu;
                    }
                    else if((fu <= fv) || (v == x) || (v == w)){
                        v = u;
                        fv = fu;
                    }
                }
                m_status = checkConvergence(stop_, current_);

                if(upper_bound < lower_bound){
                    scalar aux = lower_bound;
                    lower_bound = upper_bound;
                    upper_bound = aux;
                }
            }while(--count && m_status == status::Continue);
            return x;
        }
    };
}
#endif