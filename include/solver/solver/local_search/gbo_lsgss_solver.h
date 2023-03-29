#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_GBO_LSGSS_SOLVER_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_GBO_LSGSS_SOLVER_H

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

    static default_random_engine generator_((int) std::chrono::system_clock::now().time_since_epoch().count());

    template<typename problem_type>
    class gbo_lsgss_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::vector_t;
        using typename superclass::scalar;
        using criteria_t = typename problem_type::criteria_t;

    protected:
        deque<deque<bool>> improvement_groups;
        deque<deque<vector<scalar>>> steps;

        vector<size_t> indices_slot;
        vector<size_t> indices_step;

        size_t number_groups;
        size_t number_steps;
        scalar alpha;
        scalar beta;
        scalar initial_fit_accumulation;
        size_t eval_per_cycle;
        size_t dim;

        bool use_decomposition_functions;
        bool auto_update_steps;

        vector_t best_solution;
        vector_t fx_best_solution;

        size_t number_sub_functions;
        vector<vector<size_t>> sub_function_to_variable;
        unordered_map<size_t, std::pair<bool, size_t>> variable_to_sub_function;

    public:

        gbo_lsgss_solver() = default;

        explicit gbo_lsgss_solver(const criteria_t &s) : superclass(s) {};

        bool check_data_structure() {
            if (dim == improvement_groups.size()) {
                return true;
            } else {
                return false;
            }
        }

        void update_best_solution(vector_t &x, vector_t fx_){
            status current_status = checkConvergence(this->stop_, this->current_);
            if(current_status == status::Continue){
                if(fx_[number_sub_functions] < fx_best_solution[number_sub_functions]){
                    best_solution = x;
                    fx_best_solution = fx_;
                    this->current_.fx_best = fx_best_solution[number_sub_functions];
                    this->m_stats.push(fx_[number_sub_functions], this->current_.evaluations, this->options_.get_storage_criteria());
                }
            }
        }

        void setting_method(problem_type &problem) {
            number_groups = this->options_.get_number_groups();
            number_steps = this->options_.get_number_steps();
            beta = this->options_.get_beta();
            alpha = this->options_.get_alpha();
            initial_fit_accumulation = this->options_.get_initial_fit_accumulation();
            eval_per_cycle = this->options_.get_eval_per_cycle();
            auto_update_steps = this->options_.get_auto_update_steps();

            use_decomposition_functions = this->options_.get_use_decomposition_function();
            if(use_decomposition_functions){
                sub_function_to_variable = problem.get_sub_function_to_variable();
                variable_to_sub_function = problem.get_variable_to_sub_function();
            }
            else{
                std::cerr << "GBO-LSGSS method requires decomposition functions parameter aticved." << std::endl;
                throw 1;
            }
            number_sub_functions = sub_function_to_variable.size();

            best_solution = vector_t::Ones(dim);
            fx_best_solution = vector_t::Zero(number_sub_functions+1);
            fx_best_solution[number_sub_functions] = max_limits<scalar>();

            indices_slot = vector<size_t>(dim, 0);
            indices_step = vector<size_t>(dim, 0);
        }

        void minimize(problem_type &problem, vector_t &x) {
            dim = x.size();
            setting_method(problem);
            this->current_.reset();
            if(!problem.isValid(x))
                std::cerr << "start with invalid x0" << std::endl;
            if(!this->check_data_structure()){
                create_steps(problem, dim);
            }
            gbo_lsgss(problem, x);
        }

        vector_t get_best_solution(){
            return best_solution;
        }

    protected:
        void get_groups_to_otimizer(vector_t &fit_accumulation, vector<size_t> &groups){
            scalar max_fit = fit_accumulation[0];
            for(size_t i = 1; i < number_sub_functions; i++){
                if(fit_accumulation[i] > max_fit){
                    max_fit = fit_accumulation[i];
                }
            }
            groups.clear();
            for(int i = number_sub_functions-1; i >= 0; i--){
                if(fit_accumulation[i] == max_fit){
                    groups.push_back(i);
                }
            }
        }

        void create_steps(problem_type &problem, size_t number_sets) {
            for (size_t i = 0; i < number_sets; i++) {
                deque <vector<scalar>> step_x;
                deque<bool> slot_improvement_x;
                scalar step = alpha * (problem.upperBound()[i] - problem.lowerBound()[i]);
                for (size_t j = 0; j < number_groups; j++) {
                    vector <scalar> steps_aux;
                    for (size_t k = 0; k < number_steps; k++) {
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

        void update_steps(problem_type &problem, size_t i_set) {
            size_t i = 0;
            size_t index_slot = steps[i_set].size() - 1;
            scalar step = steps[i_set][index_slot][number_steps - 1];
            while (improvement_groups[i_set][0] == false && i < improvement_groups[i_set].size()) {
                i++;
                steps[i_set].pop_front();
                improvement_groups[i_set].pop_front();
                step = step * beta;
                vector <scalar> steps_aux;
                for (size_t k = 0; k < number_steps; k++) {
                    steps_aux.push_back(step);
                    step = step * beta;
                }
                steps[i_set].push_back(steps_aux);
                improvement_groups[i_set].push_back(false);
            }
            for (i = 0; i < improvement_groups[i_set].size(); i++) {
                improvement_groups[i_set][i] = false;
            }
        }

        scalar step_test(problem_type &problem, const vector_t &x, const vector_t &fx, const scalar step, const int index){
            vector_t trial(dim);
            for(size_t i = 0; i < dim; i++){
                trial[i] = x[i];
            }
            vector_t fx_tmp(number_sub_functions + 1);
            for(size_t i = 0; i <= number_sub_functions; i++){
                fx_tmp[i] = fx[i];
            }
            trial[index] = trial[index] + step;
            if(trial[index] > problem.upperBound()[index] || trial[index] < problem.lowerBound()[index])
                return max_limits<scalar>();
            scalar delta = problem.delta(trial, 0, trial[index], index);
            this->current_.evaluations++;
            return delta;
        }

        void lsgss(problem_type &problem, size_t i_sub_function, vector_t &x0, vector_t &fx, criteria_t &stop_cycle_criteria, criteria_t &current_cycle_criteria){
            status current_global_status = checkConvergence(this->stop_, this->current_);
            status current_cycle_status = checkConvergence(stop_cycle_criteria, current_cycle_criteria);
            vector<size_t> v(sub_function_to_variable[i_sub_function].begin(), sub_function_to_variable[i_sub_function].end());
            while(current_global_status == status::Continue && current_cycle_status == status::Continue){
                shuffle(v.begin(), v.end(), generator_);
                scalar step;
                long old_current_evaluations = this->current_.evaluations;
                for (auto i : v) {
                    step = steps[i][indices_slot[i]][indices_step[i]];
                    scalar fx_trial_1 = step_test(problem, x0, fx, step, i);
                    scalar fx_trial_2 = step_test(problem, x0, fx, -step, i);
                    scalar fx_current, fx_tmp;
                    if(number_sub_functions > 1){
                        fx_current = fx[i_sub_function];
                    }
                    else{
                        fx_current = fx[number_sub_functions];
                    }
                    bool update_fx = false;
                    if (fx_trial_1 < fx_trial_2 && fx_trial_1 < fx_current) {
                        x0[i] = x0[i] + step;
                        fx_tmp = fx_trial_1;
                        improvement_groups[i][indices_slot[i]] = true;
                        update_fx = true;
                    } else if (fx_trial_2 < fx_trial_1 && fx_trial_2 < fx_current) {
                        x0[i] = x0[i] - step;
                        fx_tmp = fx_trial_2;
                        improvement_groups[i][indices_slot[i]] = true;
                        update_fx = true;
                    }
                    if(update_fx){
                        if(number_sub_functions > 1){
                            fx[i_sub_function] = fx_tmp;
                            fx_tmp = 0.0;
                            for(size_t j = 0; j < number_sub_functions; j++){
                                fx_tmp += fx[j];
                            }
                            fx[number_sub_functions] = fx_tmp;
                        }
                        else{
                            fx[number_sub_functions] = fx_tmp;
                        }
                        update_best_solution(x0, fx);
                    }
                    indices_step[i] = ((indices_step[i] + 1) % number_steps);
                    if (indices_step[i] == 0) {
                        indices_slot[i] = ((indices_slot[i] + 1) % number_groups);
                    }
                    if (indices_step[i] == 0 && indices_slot[i] == 0 && auto_update_steps) {
                        update_steps(problem, i);
                    }
                }
                current_cycle_criteria.evaluations += (this->current_.evaluations - old_current_evaluations);
                current_global_status = checkConvergence(this->stop_, this->current_);
                current_cycle_status = checkConvergence(stop_cycle_criteria, current_cycle_criteria);
            }
        }

        void gbo_lsgss(problem_type &problem, vector_t &x){
            vector_t fx_current = vector_t::Ones(number_sub_functions+1);
            scalar fx = 0.0;
            if(use_decomposition_functions){
                for(size_t i = 0; i < number_sub_functions; i++){
                    size_t k = sub_function_to_variable[i][0];
                    fx_current[i] = problem.delta(x, 0, x[k], k);
                    ++this->current_.evaluations;
                    fx += fx_current[i];
                }
            }
            else{
                fx = problem.value(x);
                ++this->current_.evaluations;
            }
            fx_current[number_sub_functions] = fx;
            update_best_solution(x, fx_current);

            vector_t fit_accumulation = vector_t::Ones(number_sub_functions) * initial_fit_accumulation;
            vector<size_t> groups;
            groups.reserve(number_sub_functions);
            status current_status = checkConvergence(this->stop_, this->current_);
            while(current_status == status::Continue) {
                this->current_.iterations++;
                get_groups_to_otimizer(fit_accumulation, groups);
                for(size_t i = 0; i < groups.size(); i++) {
                    size_t id_group = groups[i];
                    criteria_t stop_cycle_criteria = this->stop_;
                    if(number_sub_functions > 1){
                        stop_cycle_criteria.evaluations = sub_function_to_variable[id_group].size() * eval_per_cycle;
                    }
                    else{
                        stop_cycle_criteria.evaluations = this->stop_.evaluations;
                    }
                    criteria_t current_cycle_criteria;
                    current_cycle_criteria.reset();
                    size_t index_fx;
                    if (use_decomposition_functions) {
                        index_fx = id_group;
                    } else {
                        index_fx = number_sub_functions;
                    }
                    scalar old_fx_gain = fx_current[index_fx];
                    scalar old_fx = fx_current[number_sub_functions];
                    lsgss(problem, id_group, x, fx_current, stop_cycle_criteria, current_cycle_criteria);
                    if (fx_current[number_sub_functions] < old_fx) {
                        scalar fit_improved = (old_fx - fx_current[number_sub_functions]) / old_fx;
                        fit_accumulation[id_group] = (fit_accumulation[id_group] + fit_improved) / 2.0;
                    } else {
                        fit_accumulation[id_group] = fit_accumulation[id_group] / 2.0;
                    }
                    if (this->m_debug >= debug_level::Low) {
                        cout << "Iteration: " << this->current_.iterations
                             << " - Evaluations: " << this->current_.evaluations
                             << " - f(x): " << setprecision(20) << fx_current[number_sub_functions]
                             << " - Gain: " << (old_fx_gain - fx_current[index_fx])
                             << " - ID Subproblem: " << id_group
                             << " - Fitness Improvement Accumulation: " << fit_accumulation[id_group] << endl;
                    }
                }
                current_status = checkConvergence(this->stop_, this->current_);
                this->m_status = current_status;
            }
            if(this->m_debug >= debug_level::Low) {
                cout << "Iteration: " << this->current_.iterations << " - Evaluations: "
                     << this->current_.evaluations << " - f(x): " << setprecision(20) << fx_current[number_sub_functions]
                     << endl;
            }
        }
    };
}

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_GBO_LSGSS_SOLVER_H