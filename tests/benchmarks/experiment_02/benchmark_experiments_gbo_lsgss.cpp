#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include "ThreadPool.h"
#include <solver/common.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f1_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f2_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f3_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f4_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f5_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f6_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f7_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f8_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f9_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f10_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f11_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f12_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f13_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f14_graybox_cec.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/f15_graybox_cec.h>
#include <solver/solver/local_search/lsgss_gbo_solver.h>

using namespace solver;
using namespace std;

typedef long double scalar;

class experiment {

public:
    experiment() = default;
    size_t id{};
    string exp_file_name;
    scalar beta{};
    scalar alpha{};
    size_t number_groups{};
    size_t number_steps{};
    size_t cycle{};
    scalar fit_accumulation{};
};

template <typename T>
void get_random_generate_solution(T &x, size_t id_function, size_t id_rep){
    string data_dir = "initialsolutions";
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

template<typename scalar, typename vector_t>
void save_results(const string &id_version, size_t id_func, size_t id_rep, bool is_valid_solution, vector_t &x,
                  scalar time_taken, stats<scalar> &stats_, scalar beta, scalar alpha, size_t number_groups,
                  size_t number_steps, size_t cycle, scalar fit_accumulation) {
    scalar milestone_fx_1, milestone_fx_2, milestone_fx_3;
    std::chrono::duration<scalar> milestone_time_1{}, milestone_time_2{}, milestone_time_3{};

    milestone_fx_1 = stats_.get_history()[0].fx;
    milestone_fx_2 = stats_.get_history()[1].fx;
    milestone_fx_3 = stats_.get_history()[2].fx;

    milestone_time_1 = stats_.get_history()[0].time;
    milestone_time_2 = stats_.get_history()[1].time;
    milestone_time_3 = stats_.get_history()[2].time;

    ofstream file_results;
    ofstream file_time_results;
    ofstream file_solutions;

    file_results.open(id_version + ".txt", std::ofstream::app);
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha
                 << "; " << beta << "; " << cycle << "; " << setprecision(5) << fit_accumulation << "; " << id_func << "; " << id_rep << "; 1.2e5; " << setprecision(20)
                 << milestone_fx_1 << "; " << is_valid_solution << "; " << milestone_time_1.count() << endl;
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha
                 << "; " << beta << "; " << cycle << "; " << setprecision(5) << fit_accumulation << "; " << id_func << "; " << id_rep << "; 6e5; " << setprecision(20) << milestone_fx_2
                 << "; " << is_valid_solution << "; " << milestone_time_2.count() << endl;
    file_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha
                 << "; " << beta << "; " << cycle << "; " << setprecision(5) << fit_accumulation << "; " << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3
                 << "; " << is_valid_solution << "; " << milestone_time_3.count() << endl;
    file_results.close();

    file_time_results.open(id_version + "_time.txt", std::ofstream::app);
    file_time_results << id_version << "; " << number_groups << "; " << number_steps << "; " << setprecision(5) << alpha
                      << "; " << beta << "; " << cycle << "; " << setprecision(5) << fit_accumulation << "; " << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3
                      << "; " << is_valid_solution << "; " << setprecision(20) << time_taken << endl;
    file_time_results.close();

    file_solutions.open(id_version + "_solutions.txt", std::ofstream::app);
    file_solutions << id_version << "; " << "Func: " << id_func << "; Rep: " << id_rep << "; ";
    for (int i = 0; i < x.size(); i++) {
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

    const string name_method = "lsgss_gbo_";
    const int id_version_const = 1;

    scalar beta_array[] = {0.975};
    scalar alpha_array[] = {0.25};
    scalar fit_accumulation_array[] = {1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0};
    size_t number_groups_array[] = {2};
    size_t number_steps_array[] = {2};
    size_t number_cycle_array[] = {100};

    size_t number_beta = 1;
    size_t number_alpha = 1;
    size_t number_groups = 1;
    size_t number_steps = 1;
    size_t number_cycles = 1;
    size_t number_fit_accumulation = 6;

    int id_version = id_version_const;

    for(size_t i_fit = 0; i_fit < number_fit_accumulation; i_fit++) {
        for (size_t i_cycle = 0; i_cycle < number_cycles; i_cycle++) {
            for (size_t i_group = 0; i_group < number_groups; i_group++) {
                for (size_t i_step = 0; i_step < number_steps; i_step++) {
                    for (size_t i_alpha = 0; i_alpha < number_alpha; i_alpha++) {
                        for (size_t i_beta = 0; i_beta < number_beta; i_beta++) {
                            experiment e;
                            e.number_groups = number_groups_array[i_group];
                            e.number_steps = number_steps_array[i_step];
                            e.alpha = alpha_array[i_alpha];
                            e.beta = beta_array[i_beta];
                            e.cycle = number_cycle_array[i_cycle];
                            e.fit_accumulation = fit_accumulation_array[i_fit];
                            e.exp_file_name = name_method + to_string(id_version);
                            cout << "e.exp_file_name: " << e.exp_file_name << endl;
                            experiments_.push_back(e);
                            id_version++;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < experiments_.size(); i++) {

        pool.enqueue([i, experiments_] {

        string general_results = "general_lsgss_gbo_results.txt";
        string general_results_func = "general_lsgss_gbo_results_func.txt";
        scalar mean = 0.0;
        size_t cont_exp = 0;
        scalar current_fx_best;

        typedef problem_interface<scalar>::vector_t vector_t;

        const size_t rep = 30;
        const long number_evaluations = 3e6;

        cout << "i: " << i + 1 << endl;
        cout << "name_file: " << experiments_[i].exp_file_name << endl;

        for (size_t i_function = 1; i_function <= 15; i_function++) {

            scalar mean_func = 0.0;
            size_t cont_exp_func = 0;

            size_t DIM;
            if (i_function == 13 || i_function == 14) {
                DIM = 905;
            } else {
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
                        typedef f1_graybox_cec<scalar> f1_cec;

                        f1_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);

                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);


                        lsgss_gbo_solver<f1_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 2: {
                        typedef f2_graybox_cec<scalar> f2_cec;

                        f2_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f2_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);
                        break;
                    }
                    case 3: {
                        typedef f3_graybox_cec<scalar> f3_cec;

                        f3_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f3_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 4: {
                        typedef f4_graybox_cec<scalar> f4_cec;

                        f4_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f4_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 5: {
                        typedef f5_graybox_cec<scalar> f5_cec;

                        f5_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f5_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 6: {
                        typedef f6_graybox_cec<scalar> f6_cec;

                        f6_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f6_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 7: {
                        typedef f7_graybox_cec<scalar> f7_cec;

                        f7_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f7_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 8: {
                        typedef f8_graybox_cec<scalar> f8_cec;

                        f8_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f8_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 9: {
                        typedef f9_graybox_cec<scalar> f9_cec;

                        f9_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f9_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 10: {
                        typedef f10_graybox_cec<scalar> f10_cec;

                        f10_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f10_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 11: {
                        typedef f11_graybox_cec<scalar> f11_cec;

                        f11_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f11_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 12: {
                        typedef f12_graybox_cec<scalar> f12_cec;

                        f12_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f12_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 13: {
                        typedef f13_graybox_cec<scalar> f13_cec;

                        f13_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f13_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 14: {
                        typedef f14_graybox_cec<scalar> f14_cec;

                        f14_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f14_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    case 15: {
                        typedef f15_graybox_cec<scalar> f15_cec;

                        f15_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_number_groups(experiments_[i].number_groups);
                        op.set_number_steps(experiments_[i].number_steps);
                        op.set_alpha(experiments_[i].alpha);
                        op.set_beta(experiments_[i].beta);
                        op.set_eval_per_cycle(experiments_[i].cycle);
                        op.set_initial_fit_accumulation(experiments_[i].fit_accumulation);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                        op.set_use_decomposition_function(true);

                        lsgss_gbo_solver<f15_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats(),
                                     experiments_[i].beta, experiments_[i].alpha,
                                     experiments_[i].number_groups, experiments_[i].number_steps,
                                     experiments_[i].cycle, experiments_[i].fit_accumulation);

                        break;
                    }
                    default: {
                        std::cerr << "Invalid id function" << std::endl;
                    }
                }
                mean += current_fx_best;
                cont_exp++;
                mean_func += current_fx_best;
                cont_exp_func++;
            }
            mean_func = mean_func / cont_exp_func;
            ofstream file_general_results_func;
            file_general_results_func.open(general_results_func, std::ofstream::app);
            file_general_results_func << experiments_[i].exp_file_name << "; " << experiments_[i].cycle << "; " << setprecision(5) << experiments_[i].fit_accumulation << "; 3e6; " << i_function << "; " << setprecision(20) << mean_func << "; " << setprecision(2) << cont_exp_func << "; " << endl;
            file_general_results_func.close();
        }
        mean = mean / cont_exp;
        ofstream file_general_results;
        file_general_results.open(general_results, std::ofstream::app);
        file_general_results << experiments_[i].exp_file_name << "; " << experiments_[i].cycle << "; " << setprecision(5) << experiments_[i].fit_accumulation << "; 3e6; " << setprecision(20) << mean << "; " << setprecision(2) << cont_exp << "; " << endl;
        file_general_results.close();
        });
    }
}