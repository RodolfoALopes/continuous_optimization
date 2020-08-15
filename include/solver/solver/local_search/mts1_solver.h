#ifndef CONTINUOUS_OPTIMIZATION_MTS1_SOLVER_H
#define CONTINUOUS_OPTIMIZATION_MTS1_SOLVER_H

#include <iostream>
#include <solver/solver/solver_interface.h>

using namespace std;

namespace solver {

    template<typename problem_type>
    class mts1_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::vector_t;
        using typename superclass::scalar;
        using criteria_t = typename problem_type::criteria_t;


    public:

        mts1_solver() {};

        explicit mts1_solver(const criteria_t &s) : superclass(s) {};


        void mts_ls1_improve_dim(problem_type &problem, vector_t &sol, scalar &fx, vector_t &best, scalar &fx_best, unsigned i, scalar SR){

            vector_t new_sol = sol;
            new_sol[i] -= SR;

            new_sol[i] = min(new_sol[i], problem.upperBound()[i]);
            new_sol[i] = max(new_sol[i], problem.lowerBound()[i]);

            scalar fitness_new_sol = problem.value(new_sol);
            ++this->current_.evaluations;
            this->m_stats.push(fitness_new_sol, this->current_.evaluations, this->options_.get_storage_criteria());

            if( fitness_new_sol < fx_best ){
                fx_best = fitness_new_sol;
                sol = new_sol;
            } else if( fitness_new_sol > fx_best ){
                new_sol[i] = sol[i];
                new_sol[i] += 0.5 * SR;

                new_sol[i] = min(new_sol[i], problem.upperBound()[i]);
                new_sol[i] = max(new_sol[i], problem.lowerBound()[i]);

                fitness_new_sol = problem.value(new_sol);
                ++this->current_.evaluations;
                this->m_stats.push(fitness_new_sol, this->current_.evaluations, this->options_.get_storage_criteria());

                if( fitness_new_sol < fx_best ){
                    fx_best = fitness_new_sol;
                    sol = new_sol;
                }
            }

            best = sol;

        }

        void minimize(problem_type &problem, vector_t &x0) {

            this->current_.reset();

            if(!problem.isValid(x0))
                cerr << "Start with invalid x0." << endl;


            size_t DIM = x0.size();

            scalar fx = problem.value(x0);
            ++this->current_.evaluations;
            this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

            vector_t best = x0;
            scalar fx_best = fx;

            if(this->m_debug >= debug_level::Low) {
                cout << "Iterations: " << this->current_.iterations << " - Evaluations: "
                     << this->current_.evaluations << " - f(x):" << fx_best << endl;
            }

            scalar improve;

            vector<scalar> SR(DIM);
            vector<scalar> improvement(DIM, 0.0);
            vector<scalar> dim_sorted(DIM);

            iota(dim_sorted.begin(), dim_sorted.end(), 0);

            for(size_t i = 0; i < DIM; i++){
                SR[i] = (problem.upperBound()[i] - problem.lowerBound()[i]) * 0.4;
            }


            this->m_status = checkConvergence(this->stop_, this->current_);
            if(this->m_status == status::Continue){
                next_permutation(dim_sorted.begin(), dim_sorted.end());
                for(auto it = dim_sorted.begin(); it != dim_sorted.end(); it++ ){

                    mts_ls1_improve_dim(problem, x0, fx, best, fx_best, *it, SR[*it]);

                    improve = max(fx_best - fx, (scalar)0.0);
                    improvement[*it] = improve;

                    if(improve > 0.0){
                        best = x0;
                        fx_best = fx;
                    } else {
                        SR[*it] /= 2.0f;
                    }
                }
            }

            iota(dim_sorted.begin(), dim_sorted.end(), 0);
            sort(dim_sorted.begin(), dim_sorted.end(), [&](unsigned i1, unsigned i2) { return improvement[i1] > improvement[i2]; });

            size_t i, d = 0, next_d, next_i;

            this->m_status = checkConvergence(this->stop_, this->current_);
            while(problem.callback(this->current_, x0) && (this->m_status == status::Continue)){

                i = dim_sorted[d];
                mts_ls1_improve_dim(problem, x0, fx, best, fx_best, i, SR[i]);

                improve = max(fx_best - fx, (scalar)0.0);
                improvement[i] = improve;
                next_d = (d+1)%DIM;
                next_i = dim_sorted[next_d];

                if(improve > 0.0){
                    best = x0;

                    if(improvement[i] < improvement[next_i]){
                        iota(dim_sorted.begin(), dim_sorted.end(), 0);
                        sort(dim_sorted.begin(), dim_sorted.end(), [&](unsigned i1, unsigned i2) { return improvement[i1] > improvement[i2]; });
                    }
                }
                else {
                    SR[i] /= 2.0;
                    d = next_d;
                    if(SR[i] < 1e-15){
                        SR[i] = (problem.upperBound()[i] - problem.lowerBound()[i]) * 0.4;
                    }
                }

                ++this->current_.iterations;
                this->current_.fx_best = fx_best;
                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    cout << "Iterations: " << this->current_.iterations << " - Evaluations: "
                         << this->current_.evaluations << " - f(x):" << this->current_.fx_best << endl;
                }
            }
            x0 = best;
        }
    };
}

#endif