#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>
#include "ThreadPool.h"
#include <solver/common.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f1_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f2_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f3_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f4_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f5_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f6_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f7_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f8_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f9_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f10_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f11_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f12_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f13_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f14_bbo_cec_2013.h>
#include <solver/problem/bounded_problems/bbo_cec_2013_benchmark/f15_bbo_cec_2013.h>
#include <solver/solver/local_search/lsgss_solver.h>

using namespace solver;
using namespace std;

typedef long double scalar;

class experiment{

public:
    experiment() = default;
    size_t id;
    string exp_file_name;
    scalar beta;
    scalar alpha;
    size_t number_groups;
    size_t number_steps;
};


template <typename T>
void get_random_generate_solution(T &x, size_t id_function, size_t id_rep){
    string data_dir = "initialsolutioncec2020";

    stringstream ss;
    ss << data_dir <<"/" << "solution_x0_f" << id_function << "_" << id_rep << ".txt";
    ifstream file(ss.str());
    int i = 0;
    while(!file.eof()){
        file >> x[i];
        i++;
    }
    file.close();
}

template <typename scalar, typename vector_t>
void save_results(const string &id_version, size_t id_func, size_t id_rep, bool is_valid_solution, vector_t &x, stats<scalar> &stats_,
                  scalar beta, scalar alpha, size_t number_groups, size_t number_steps){
    scalar milestone_fx_1, milestone_fx_2, milestone_fx_3;
    std::chrono::duration<scalar> milestone_time_1, milestone_time_2, milestone_time_3;

    milestone_fx_1 = stats_.get_history()[0].fx;
    milestone_fx_2 = stats_.get_history()[1].fx;
    milestone_fx_3 = stats_.get_history()[2].fx;

    milestone_time_1 = stats_.get_history()[0].time;
    milestone_time_2 = stats_.get_history()[1].time;
    milestone_time_3 = stats_.get_history()[2].time;

    ofstream file_results;
    ofstream file_solutions;

    file_results.open(id_version + ".txt", std::ofstream::app);
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha << "; " << beta << "; " << id_func << "; " << id_rep << "; 1.2e5; " << setprecision(20) << milestone_fx_1 << "; " << is_valid_solution << "; " << milestone_time_1.count() << endl;
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha << "; " << beta << "; " << id_func << "; " << id_rep << "; 6e5; " << setprecision(20) << milestone_fx_2 << "; " << is_valid_solution << "; " << milestone_time_2.count() << endl;
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha << "; " << beta << "; " << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3 << "; " << is_valid_solution << "; " << milestone_time_3.count() << endl;
    file_results.close();

    file_solutions.open(id_version + "_solutions.txt", std::ofstream::app);
    file_solutions << id_version << "; " << "Func: " << id_func << "; Rep: " << id_rep << "; ";
    for(int i = 0; i < x.size(); i++){
        file_solutions << x[i] << "; ";
    }
    file_solutions << endl;
    file_solutions.close();

    cout << "id_version: " << id_version << endl;
    cout << "stop: 1.2e5 - fx: " << milestone_fx_1 << " - time: " << milestone_time_1.count() << endl;
    cout << "stop: 6e5 - fx: " << milestone_fx_2 << " - time: " << milestone_time_2.count() << endl;
    cout << "stop: 3e6 - fx: " << milestone_fx_3 << " - time: " << milestone_time_3.count() << endl;
    cout << "is_valid_solution(): " << is_valid_solution << endl << endl;
}


int main() {

    ThreadPool pool(1);

    vector<experiment> experiments_;

    const string name_method = "lsgss_method";
    const int id_version_const = 1;

    scalar beta_array[] = {0.975, 0.95, 0.925, 0.9, 0.875, 0.85};
    scalar alpha_array[] = {0.45, 0.35, 0.25};
    size_t number_groups_array[] = {2, 4, 6};
    size_t number_steps_array[] = {2, 5, 10};

    size_t number_beta = 6;
    size_t number_alpha = 3;
    size_t number_groups = 3;
    size_t number_steps = 3;

    int id_version = id_version_const;

    for(size_t i_slot = 0; i_slot < number_groups; i_slot++){
        for(size_t i_step = 0; i_step < number_steps; i_step++){
            for(size_t i_alpha = 0; i_alpha < number_alpha; i_alpha++){
                for(size_t i_beta = 0; i_beta < number_beta; i_beta++){
                    experiment e;
                    e.number_groups = number_groups_array[i_slot];
                    e.number_steps = number_steps_array[i_step];
                    e.alpha = alpha_array[i_alpha];
                    e.beta = beta_array[i_beta];
                    e.exp_file_name = name_method + "_" + to_string(id_version);
                    cout << "e.exp_file_name: " << e.exp_file_name << endl;
                    experiments_.push_back(e);
                    id_version++;
                }
            }
        }
    }

    for(int i = 0; i < experiments_.size(); i++){
        pool.enqueue([i, experiments_] {

            typedef problem_interface<scalar>::vector_t vector_t;

            const size_t rep = 30;
            const long number_evaluations = 3e6;

            cout << "i: " << i + 1 << endl;
            cout << "name_file: " << experiments_[i].exp_file_name << endl;

            for (size_t i_function = 1; i_function <= 15; i_function++) {

                size_t DIM;
                if(i_function == 13 || i_function == 14){
                    DIM = 905;
                }
                else{
                    DIM = 1000;
                }

                for (size_t i_rep = 1; i_rep <= rep; i_rep++) {

                    cout << "ID_FUNC: " << i_function << " - ID_REP: " << i_rep << endl;
                    solver::criteria<scalar> criteria_ = solver::criteria<scalar>::defaults();
                    criteria_.iterations = LONG_MAX;
                    criteria_.evaluations = number_evaluations;
                    criteria_.gradNorm = -0.1;
                    criteria_.fx_is_know = true;
                    criteria_.fx_best = 0.0;
                    criteria_.error_fx_best = 0.0;

                    switch (i_function) {
                        case 1: {
                            typedef f1_bbo_cec_2013<scalar> f1_cec;

                            f1_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);

                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f1_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 2: {
                            typedef f2_bbo_cec_2013<scalar> f2_cec;

                            f2_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f2_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 3: {
                            typedef f3_bbo_cec_2013<scalar> f3_cec;

                            f3_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f3_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 4: {
                            typedef f4_bbo_cec_2013<scalar> f4_cec;

                            f4_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f4_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 5: {
                            typedef f5_bbo_cec_2013<scalar> f5_cec;

                            f5_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f5_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 6: {
                            typedef f6_bbo_cec_2013<scalar> f6_cec;

                            f6_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f6_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 7: {
                            typedef f7_bbo_cec_2013<scalar> f7_cec;

                            f7_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f7_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 8: {
                            typedef f8_bbo_cec_2013<scalar> f8_cec;

                            f8_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f8_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 9: {
                            typedef f9_bbo_cec_2013<scalar> f9_cec;

                            f9_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f9_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 10: {
                            typedef f10_bbo_cec_2013<scalar> f10_cec;

                            f10_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f10_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 11: {
                            typedef f11_bbo_cec_2013<scalar> f11_cec;

                            f11_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f11_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 12: {
                            typedef f12_bbo_cec_2013<scalar> f12_cec;

                            f12_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f12_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 13: {
                            typedef f13_bbo_cec_2013<scalar> f13_cec;

                            f13_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f13_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 14: {
                            typedef f14_bbo_cec_2013<scalar> f14_cec;

                            f14_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f14_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        case 15: {
                            typedef f15_bbo_cec_2013<scalar> f15_cec;

                            f15_cec f(DIM);

                            vector_t x0 = vector_t::Ones(DIM);
                            get_random_generate_solution(x0, i_function, i_rep);

                            options<scalar> op = options<scalar>::defaults();
                            op.set_number_groups(experiments_[i].number_groups);
                            op.set_number_steps(experiments_[i].number_steps);
                            op.set_alpha(experiments_[i].alpha);
                            op.set_beta(experiments_[i].beta);
                            op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                            lsgss_solver<f15_cec> solver;
                            solver.set_options(op);
                            solver.set_stop_criteria(criteria_);
                            solver.minimize(f, x0);
                            bool is_valid_solution = f.isValid(x0);

                            save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats(),
                                         experiments_[i].beta, experiments_[i].alpha,
                                         experiments_[i].number_groups, experiments_[i].number_steps);

                            break;
                        }
                        default:{
                            std::cerr << "Invalid id function" << std::endl;
                        }
                    }

                }
            }
        });
    }
}