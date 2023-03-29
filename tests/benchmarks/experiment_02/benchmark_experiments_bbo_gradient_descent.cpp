#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
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
#include <solver/solver/local_search/gradient_descent_solver.h>
#include <iomanip>


using namespace solver;
using namespace std;

typedef long double scalar;


template<typename T>
void get_random_generate_solution(T &x, size_t id_function, size_t id_rep) {
    string data_dir = "initialsolutions";
    stringstream ss;
    ss << data_dir << "/" << "solution_x0_f" << id_function << "_" << id_rep << ".txt";
    ifstream file(ss.str());
    int i = 0;
    while (!file.eof()) {
        file >> x[i];
        i++;
    }
    file.close();
}

template<typename scalar, typename vector_t>
void save_results(const string &id_version, size_t id_func, size_t id_rep, bool is_valid_solution, vector_t &x,
                  scalar time_taken, stats<scalar> &stats_) {
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
    file_results << id_version << "; " << id_func << "; " << id_rep << "; 1.2e5; " << setprecision(20)
                 << milestone_fx_1 << "; " << is_valid_solution << "; " << milestone_time_1.count() << endl;
    file_results << id_version << "; "  << "; " << id_func << "; " << id_rep << "; 6e5; " << setprecision(20) << milestone_fx_2
                 << "; " << is_valid_solution << "; " << milestone_time_2.count() << endl;
    file_results << id_version << "; " << "; " << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3
                 << "; " << is_valid_solution << "; " << milestone_time_3.count() << endl;
    file_results.close();

    file_time_results.open(id_version + "_time.txt", std::ofstream::app);
    file_time_results << id_version << "; " << "; " << id_func << "; " << id_rep << "; 3e6; " << setprecision(20) << milestone_fx_3
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

    const string name_method = "gd_bbo";
    const int id_version_const = 1;
    const scalar tol_line_search = 1e-20;

    string general_results = "general_gd_bbo_results.txt";
    string general_results_func = "general_gd_bbo_results_func.txt";
    scalar mean = 0.0;
    size_t cont_exp = 0;
    scalar current_fx_best;

    typedef problem_interface<scalar>::vector_t vector_t;

    const size_t rep = 30;
    const long number_evaluations = 3e6;


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
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f1_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 2: {
                        typedef f2_graybox_cec<scalar> f2_cec;

                        f2_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f2_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 3: {
                        typedef f3_graybox_cec<scalar> f3_cec;

                        f3_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f3_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 4: {
                        typedef f4_graybox_cec<scalar> f4_cec;

                        f4_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f4_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 5: {
                        typedef f5_graybox_cec<scalar> f5_cec;

                        f5_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f5_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 6: {
                        typedef f6_graybox_cec<scalar> f6_cec;

                        f6_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f6_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 7: {
                        typedef f7_graybox_cec<scalar> f7_cec;

                        f7_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f7_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 8: {
                        typedef f8_graybox_cec<scalar> f8_cec;

                        f8_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f8_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 9: {
                        typedef f9_graybox_cec<scalar> f9_cec;

                        f9_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f9_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 10: {
                        typedef f10_graybox_cec<scalar> f10_cec;

                        f10_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f10_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);

                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 11: {
                        typedef f11_graybox_cec<scalar> f11_cec;

                        f11_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f11_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 12: {
                        typedef f12_graybox_cec<scalar> f12_cec;

                        f12_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f12_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 13: {
                        typedef f13_graybox_cec<scalar> f13_cec;

                        f13_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f13_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 14: {
                        typedef f14_graybox_cec<scalar> f14_cec;

                        f14_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f14_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

                        break;
                    }
                    case 15: {
                        typedef f15_graybox_cec<scalar> f15_cec;

                        f15_cec f(DIM);

                        vector_t x0 = vector_t::Ones(DIM);
                        get_random_generate_solution(x0, i_function, i_rep);

                        options<scalar> op = options<scalar>::defaults();
                        op.set_line_search(line_search::Brent);
                        op.set_line_search_tolerance(tol_line_search);
                        op.set_storage_criteria(storage_criteria_level::CEC_Stop_Points);

                        gradient_descent_solver<f15_cec> solver;
                        solver.set_options(op);
                        solver.set_stop_criteria(criteria_);


                        auto start = chrono::high_resolution_clock::now();
                        ios_base::sync_with_stdio(false);
                        solver.minimize(f, x0);
                        auto end = chrono::high_resolution_clock::now();
                        scalar time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
                        time_taken *= 1e-9;

                        bool is_valid_solution = f.isValid(x0);

                        save_results(name_method, i_function, i_rep, is_valid_solution, x0,
                                     time_taken, solver.get_stats());

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
            file_general_results_func << name_method << "; 3e6; " << i_function << "; " << setprecision(20) << mean_func << "; " << setprecision(2) << cont_exp_func << "; " << endl;
            file_general_results_func.close();
        }
        mean = mean / cont_exp;
        ofstream file_general_results;
        file_general_results.open(general_results, std::ofstream::app);
        file_general_results << name_method << "; 3e6; " << setprecision(20) << mean << "; " << setprecision(2) << cont_exp << "; " << endl;
        file_general_results.close();
}