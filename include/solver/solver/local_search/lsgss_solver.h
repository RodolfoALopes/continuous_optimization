#ifndef CONTINUOUS_OPTIMIZATION_LSGSS_SOLVER_H
#define CONTINUOUS_OPTIMIZATION_LSGSS_SOLVER_H

#include <iostream>
#include <algorithm>
#include <random>
#include <vector>
#include <list>
#include <set>
#include <deque>
#include <chrono>
#include <iomanip>
#include <solver/solver/solver_interface.h>

using namespace std;

namespace solver {

    static default_random_engine generator_ ((int) std::chrono::system_clock::now().time_since_epoch().count());

    template<typename problem_type>
    class lsgss_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::vector_t;
        using typename superclass::scalar;
        using criteria_t = typename problem_type::criteria_t;

    protected:
        deque<deque<bool>> improvement_groups;
        deque<deque<vector<scalar>>> steps;

        size_t number_groups;
        size_t number_steps;
        scalar alpha;
        scalar beta;
        size_t dim;
        bool use_decomposition_functions;
        bool auto_update_steps;

    public:

        lsgss_solver() = default;

        explicit lsgss_solver(const criteria_t &s) : superclass(s) {};

        bool check_data_structure(){
            return dim == improvement_groups.size() ? true : false;
        }

        void setting_method(){
            number_groups = this->options_.get_number_groups();
            number_steps = this->options_.get_number_steps();
            beta = 1 - this->options_.get_beta();
            alpha = this->options_.get_alpha();
            use_decomposition_functions = this->options_.get_use_decomposition_function();
            auto_update_steps = this->options_.get_auto_update_steps();
        }

        void create_steps(problem_type &problem, size_t number_sets){
            this->setting_method();
            for(size_t i = 0; i < number_sets; i++){
                deque<vector<scalar>> step_x;
                deque<bool> slot_improvement_x;
                scalar step = alpha * (problem.upperBound()[i] - problem.lowerBound()[i]);
                for(size_t j = 0; j < number_groups; j++){
                    vector<scalar> steps_aux;
                    for(size_t k = 0; k < number_steps; k++){
                        steps_aux.push_back(step);
                        step = step * beta;
                    }
                    step_x.push_back(steps_aux);
                    slot_improvement_x.push_back(false);
                }
                steps.push_back(step_x);
                improvement_groups.push_back(slot_improvement_x);
            }
        }

        void update_steps(size_t i_set){
            size_t i = 0;
            size_t index_slot = steps[i_set].size()-1;
            scalar step = steps[i_set][index_slot][number_steps-1];

            while(improvement_groups[i_set][0] == false && i < improvement_groups[i_set].size()){
                i++;

                steps[i_set].pop_front();
                improvement_groups[i_set].pop_front();

                step = step * beta;

                vector<scalar> steps_aux;
                for(size_t k = 0; k < number_steps; k++){
                    steps_aux.push_back(step);
                    step = step * beta;
                }
                steps[i_set].push_back(steps_aux);
                improvement_groups[i_set].push_back(false);
            }

            for(i = 0; i < improvement_groups[i_set].size(); i++){
                improvement_groups[i_set][i] = false;
            }
        }


        scalar step_test(const vector_t &x, problem_type &problem, scalar step, int index, criteria<scalar> &current_criteria,
                         stats<scalar> &current_data_stats){

            vector_t trial(dim);
            for(int i = 0; i < dim; i++){
                trial[i] = x[i];
            }
            trial[index] = trial[index] + step;
            if(trial[index] > problem.upperBound()[index] || trial[index] < problem.lowerBound()[index])
                return max_limits<scalar>();
            scalar fx = problem.value(trial);
            ++current_criteria.evaluations;
            current_data_stats.push(fx, current_criteria.evaluations, this->options_.get_storage_criteria());
            return fx;

        }

        void lsgss(problem_type &problem, vector_t &x0, scalar &fx, criteria<scalar> &stop_criteria, criteria<scalar> &current_criteria,
                status &current_status, stats<scalar> &current_data_stats){

            vector<size_t> indices_slot(dim, 0);
            vector<size_t> indices_step(dim, 0);

            vector<vector<size_t>> sub_function_to_variable;

            if(use_decomposition_functions){
                sub_function_to_variable = problem.get_sub_function_to_variable();
            }
            else{
                vector<size_t> v(dim);
                iota(v.begin(), v.end(), 0);
                sub_function_to_variable.push_back(v);
            }

            current_status = checkConvergence(stop_criteria, current_criteria);

            if(this->m_debug >= debug_level::Low) {
                cout << "Iteration: " << current_criteria.iterations << " - Evaluations: "
                     << current_criteria.evaluations << " - f(x):" << current_criteria.fx_best << endl;
            }

            while(current_status == status::Continue){

                for(size_t i_sub_problem = 0; i_sub_problem < sub_function_to_variable.size(); i_sub_problem++){
                    shuffle(sub_function_to_variable[i_sub_problem].begin(), sub_function_to_variable[i_sub_problem].end(), generator_);

                    scalar step;

                    for (auto i : sub_function_to_variable[i_sub_problem]) {

                        step = steps[i][indices_slot[i]][indices_step[i]];

                        scalar fx_trial_1 = step_test(x0, problem, step, i, current_criteria, current_data_stats);

                        scalar fx_trial_2 = step_test(x0, problem, -step, i, current_criteria, current_data_stats);

                        if (fx_trial_1 < fx_trial_2 && fx_trial_1 < fx) {

                            x0[i] = x0[i] + step;
                            fx = fx_trial_1;

                            improvement_groups[i][indices_slot[i]] = true;

                        } else if (fx_trial_2 < fx_trial_1 && fx_trial_2 < fx) {

                            x0[i] = x0[i] - step;
                            fx = fx_trial_2;

                            improvement_groups[i][indices_slot[i]] = true;

                        }

                        indices_step[i] = ((indices_step[i] + 1) % number_steps);

                        if (indices_step[i] == 0) {
                            indices_slot[i] = ((indices_slot[i] + 1) % number_groups);
                        }

                        if (indices_step[i] == 0 && indices_slot[i] == 0 && auto_update_steps) {
                            update_steps(i);
                        }

                    }
                }

                ++current_criteria.iterations;
                current_criteria.fx_best = fx;

                if(this->m_debug >= debug_level::Low) {
                    cout << "Iterations: " << current_criteria.iterations << " - Evaluations: "
                         << current_criteria.evaluations << " - f(x):" << current_criteria.fx_best << endl;
                }
                current_status = checkConvergence(stop_criteria, current_criteria);
            }
        }

        void minimize(problem_type &problem, vector_t &x0, scalar &fx, criteria<scalar> &current_criteria, stats<scalar> &current_data_stats) {
            dim = x0.size();
            this->setting_method();

            if(!problem.isValid(x0))
                cerr << "Start with invalid x0." << endl;

            if(!this->check_data_structure()){
                create_steps(problem, dim);
            }

            this->current_.reset();
            this->current_.evaluations = current_criteria.evaluations;

            l4s(problem, x0, fx, this->stop_, this->current_, this->m_status, current_data_stats);

            current_criteria.evaluations = this->current_.evaluations;
        }

        void minimize(problem_type &problem, vector_t &x0) {
            dim = x0.size();
            this->setting_method();

            if(!problem.isValid(x0))
                cerr << "Start with invalid x0." << endl;

            if(!this->check_data_structure()){
                create_steps(problem, dim);
            }

            this->current_.reset();

            scalar fx = problem.value(x0);
            ++this->current_.evaluations;
            this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());
            this->current_.fx_best = fx;
            lsgss(problem, x0, fx, this->stop_, this->current_, this->m_status, this->m_stats);
        }
    };
}
#endif