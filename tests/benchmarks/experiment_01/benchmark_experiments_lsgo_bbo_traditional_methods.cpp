#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <random>
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
#include <solver/solver/local_search/bfgs_solver.h>
#include <solver/solver/local_search/conjugated_gradient_descent_solver.h>
#include <solver/solver/local_search/coordinate_descent_solver.h>
#include <solver/solver/local_search/gradient_descent_solver.h>
#include <solver/solver/local_search/mts1_solver.h>
#include <solver/solver/local_search/powell_solver.h>

using namespace solver;
using namespace std;

enum enum_method{GRADIENT_DESCENT, CONJUGATED_GRADIENT_DESCENT, BFGS, COORDINATE_DESCENT, POWELL_METHOD, MTS1, LAST};

enum enum_line_search{ACCELERATED_STEP_SIZE, BRENT, GOLDEN_SECTION, INTERVAL_HALVING, NON};

class experiment{

public:
    experiment() = default;
    enum_method exp_method;
    enum_line_search exp_line_search;
    string exp_file_name;
};

template <typename T>
void get_random_generate_solution(T &x, size_t id_function, size_t id_rep){
    default_random_engine generator_((int) std::chrono::system_clock::now().time_since_epoch().count());
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
void save_results(string file_name, size_t id_func, size_t id_rep, bool is_valid_solution, vector_t &x, stats<scalar> &stats_){
    scalar milestone_fx_1, milestone_fx_2, milestone_fx_3;
    std::chrono::duration<double> milestone_time_1, milestone_time_2, milestone_time_3;

    milestone_fx_1 = stats_.get_history()[0].fx;
    milestone_fx_2 = stats_.get_history()[1].fx;
    milestone_fx_3 = stats_.get_history()[2].fx;

    milestone_time_1 = stats_.get_history()[0].time;
    milestone_time_2 = stats_.get_history()[1].time;
    milestone_time_3 = stats_.get_history()[2].time;

    ofstream file_results;
    ofstream file_solutions;

    file_results.open(file_name + ".txt", std::ofstream::app);
    file_results << id_func << "; " << id_rep << "; 1.2e5; " << setprecision(20) << milestone_fx_1 << "; " << is_valid_solution << "; " << milestone_time_1.count() << endl;
    file_results << id_func << "; " << id_rep << "; 6e5; " << setprecision(20) << milestone_fx_2 << "; " << is_valid_solution << "; " << milestone_time_2.count() << endl;
    file_results << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3 << "; " << is_valid_solution << "; " << milestone_time_3.count() << endl;
    file_results.close();

    file_solutions.open(file_name + "_solutions.txt", std::ofstream::app);
    file_solutions << "Func: " << id_func << "; Rep: " << id_rep << "; ";
    for(size_t i = 0; i < x.size(); i++){
        file_solutions << x[i] << "; ";
    }
    file_solutions << endl;
    file_solutions.close();

    cout << "stop: 1.2e5 - fx: " << milestone_fx_1 << " - time: " << milestone_time_1.count() << endl;
    cout << "stop: 6e5 - fx: " << milestone_fx_2 << " - time: " << milestone_time_2.count() << endl;
    cout << "stop: 3e6 - fx: " << milestone_fx_3 << " - time: " << milestone_time_3.count() << endl;
    cout << "is_valid_solution(): " << is_valid_solution << endl << endl;
}

int main() {

    ThreadPool pool(1);
    std::vector<std::future<bool>> results;

    vector<experiment> experiments_;

    for(int i_method = enum_method::GRADIENT_DESCENT; i_method != enum_method::LAST; i_method++){

        for (int i_line_search = enum_line_search::ACCELERATED_STEP_SIZE; i_line_search < enum_line_search::NON; i_line_search++) {
            experiment e;
            string name_method = "";
            string name_line_search = "";

            e.exp_method = static_cast<enum_method>(i_method);

            if(i_method == enum_method::MTS1){
                i_line_search = enum_line_search::NON;
                e.exp_line_search = enum_line_search::NON;
            }
            else{
                switch(i_line_search){
                    case enum_line_search::ACCELERATED_STEP_SIZE:{
                        e.exp_line_search = enum_line_search::ACCELERATED_STEP_SIZE;
                        name_line_search = "accelerated_step_size";
                        break;
                    }
                    case enum_line_search::BRENT:{
                        e.exp_line_search = enum_line_search::BRENT;
                        name_line_search = "brent";
                        break;
                    }
                    case enum_line_search::GOLDEN_SECTION:{
                        e.exp_line_search = enum_line_search::GOLDEN_SECTION;
                        name_line_search = "golden_section";
                        break;
                    }
                    case enum_line_search::INTERVAL_HALVING:{
                        e.exp_line_search = enum_line_search::INTERVAL_HALVING;
                        name_line_search = "interval_halving";
                        break;
                    }
                    default:{
                        cerr << "Invalid line search algorithm." << endl;
                    }
                }
            }

            switch (i_method) {
                case enum_method::GRADIENT_DESCENT: {
                    name_method = "gradient_descent_";
                    break;
                }
                case enum_method::CONJUGATED_GRADIENT_DESCENT: {
                    name_method = "conjugated_gradient_descent_";
                    break;
                }
                case enum_method::COORDINATE_DESCENT: {
                    name_method = "coordinate_descent_";
                    break;
                }
                case enum_method::POWELL_METHOD: {
                    name_method = "powell_method_";
                    break;
                }
                case enum_method::BFGS:{
                    name_method = "bfgs_";
                    break;
                }
                case enum_method::MTS1:{
                    name_method = "mts1";
                    break;
                }
                default:{
                    cerr << "Invalid local search algorithm" << endl;
                }
            }

            e.exp_file_name = "results_" + name_method + name_line_search;

            cout << "e.exp_file_name: " << e.exp_file_name << endl;

            experiments_.push_back(e);
        }
    }

    for(int i = 0; i < experiments_.size(); i++){
        pool.enqueue([i, experiments_]{

            typedef long double scalar;
            typedef problem_interface<scalar>::vector_t vector_t;

//            const size_t rep = 30;
//            const long number_evaluations = 3e6;
            const scalar tol_line_search = 1e-20;

            const long number_evaluations = 1000;
            const size_t rep = 1;

            cout << "i: " << i+1 << endl;
            cout << "name_file: " << experiments_[i].exp_file_name << endl;

            for(size_t i_function = 1; i_function <= 15; i_function++){

                size_t DIM;
                if(i_function == 13 || i_function == 14){
                    DIM = 905;
                }
                else{
                    DIM = 1000;
                }

                for(size_t i_rep = 1; i_rep <= rep; i_rep++){

                    cout << "ID_FUNC: " << i_function << " - ID_REP: " << i_rep << endl;
                    solver::criteria<scalar> criteria_ = solver::criteria<scalar>::defaults();
                    criteria_.iterations = LONG_MAX;
                    criteria_.evaluations = number_evaluations;
                    criteria_.gradNorm = -0.1;
                    criteria_.fx_is_know = true;
                    criteria_.fx_best = 0.0;
                    criteria_.error_fx_best = 0.0;

                    switch(experiments_[i].exp_method){
                        case enum_method::CONJUGATED_GRADIENT_DESCENT :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }


                                    conjugated_gradient_descent_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);


                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search) {
                                        case enum_line_search::ACCELERATED_STEP_SIZE: {
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT: {
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION: {
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING: {
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    conjugated_gradient_descent_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        case enum_method::GRADIENT_DESCENT :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    gradient_descent_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        case enum_method::BFGS :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    bfgs_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        case enum_method::COORDINATE_DESCENT :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    coordinate_descent_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        case enum_method::POWELL_METHOD :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();
                                    switch(experiments_[i].exp_line_search){
                                        case enum_line_search::ACCELERATED_STEP_SIZE:{
                                            op.set_line_search(line_search::AcceleratedStepSize);
                                            op.set_alpha_initial(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::BRENT:{
                                            op.set_line_search(line_search::Brent);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::GOLDEN_SECTION:{
                                            op.set_line_search(line_search::GoldenSection);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        case enum_line_search::INTERVAL_HALVING:{
                                            op.set_line_search(line_search::IntervalHalving);
                                            op.set_line_search_tolerance(tol_line_search);
                                            break;
                                        }
                                        default:{
                                            throw std::logic_error("Invalid line search algorithm");
                                        }
                                    }

                                    powell_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        case enum_method::MTS1 :{
                            switch (i_function){
                                case 1:{
                                    typedef f1_bbo_cec_2013<scalar> f1_cec;

                                    f1_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f1_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 2:{
                                    typedef f2_bbo_cec_2013<scalar> f2_cec;

                                    f2_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f2_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 3:{
                                    typedef f3_bbo_cec_2013<scalar> f3_cec;

                                    f3_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f3_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 4:{
                                    typedef f4_bbo_cec_2013<scalar> f4_cec;

                                    f4_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f4_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 5:{
                                    typedef f5_bbo_cec_2013<scalar> f5_cec;

                                    f5_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f5_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 6:{
                                    typedef f6_bbo_cec_2013<scalar> f6_cec;

                                    f6_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f6_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 7:{
                                    typedef f7_bbo_cec_2013<scalar> f7_cec;

                                    f7_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f7_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 8:{
                                    typedef f8_bbo_cec_2013<scalar> f8_cec;

                                    f8_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f8_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 9:{
                                    typedef f9_bbo_cec_2013<scalar> f9_cec;

                                    f9_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f9_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 10:{
                                    typedef f10_bbo_cec_2013<scalar> f10_cec;

                                    f10_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f10_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 11:{
                                    typedef f11_bbo_cec_2013<scalar> f11_cec;

                                    f11_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f11_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 12:{
                                    typedef f12_bbo_cec_2013<scalar> f12_cec;

                                    f12_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f12_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 13:{
                                    typedef f13_bbo_cec_2013<scalar> f13_cec;

                                    f13_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f13_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 14:{
                                    typedef f14_bbo_cec_2013<scalar> f14_cec;

                                    f14_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f14_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                case 15:{
                                    typedef f15_bbo_cec_2013<scalar> f15_cec;

                                    f15_cec f(DIM);

                                    vector_t x0 = vector_t::Ones(DIM);
                                    get_random_generate_solution(x0, i_function, i_rep);

                                    options<scalar> op = options<scalar>::defaults();

                                    mts1_solver<f15_cec> solver;
                                    solver.set_options(op);
                                    solver.set_stop_criteria(criteria_);
                                    solver.minimize(f, x0);
                                    bool is_valid_solution = f.isValid(x0);

                                    save_results(experiments_[i].exp_file_name, i_function, i_rep, is_valid_solution, x0, solver.get_stats());

                                    break;
                                }
                                default:
                                    cerr << "Invalid idfunction." << endl;
                            }
                            break;
                        }
                        default:{
                            throw std::logic_error("Invalid line search algorithm");
                        }
                    }
                }
            }
        });
    }

    return 0;
}
