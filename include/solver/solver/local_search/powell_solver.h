#ifndef CONTINUOUS_OPTIMIZATION_POWELL_METHOD_H
#define CONTINUOUS_OPTIMIZATION_POWELL_METHOD_H


#include <solver/solver/solver_interface.h>
#include <solver/problem/bounded_problem_interface.h>
#include <solver/solver/line_search/accelerated_step_size.h>
#include <solver/solver/line_search/interval_halving.h>
#include <solver/solver/line_search/golden_section.h>
#include <solver/solver/line_search/brent.h>


namespace solver{

    template<typename problem_type>
    class powell_solver : public solver_interface<problem_type, 1> {

    public:
        using superclass = solver_interface<problem_type, 1>;
        using typename superclass::vector_t;
        using typename superclass::scalar;
        using criteria_t = typename problem_type::criteria_t;
        using matrix_type = Eigen::Matrix<scalar, problem_type::Dim, Eigen::Dynamic>;


    public:

        powell_solver(){}

        void apply_line_search(vector_t &x0, vector_t &s_i, problem_type &problem, scalar &fx){
            scalar rate;
            switch (this->options_.get_line_search()){
                case line_search::AcceleratedStepSize : {
                    rate = accelerated_step_size<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i),
                            this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial());
                    scalar new_fx = problem.value(x0 + rate * s_i);
                    ++this->current_.evaluations;
                    this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                    if(new_fx < fx){
                        x0 = x0 + rate * s_i;
                        fx = new_fx;
                    }else{
                        s_i *= -1;
                        rate = accelerated_step_size<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_alpha_initial());
                        new_fx = problem.value(x0 + rate * s_i);
                        ++this->current_.evaluations;
                        this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                        if(new_fx < fx){
                            x0 = x0 + rate * s_i;
                            fx = new_fx;
                        }
                    }
                    break;
                }
                case line_search::Brent : {
                    rate = brent<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i), this->stop_,
                            this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                    scalar new_fx = problem.value(x0 + rate * s_i);
                    ++this->current_.evaluations;
                    this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                    if(new_fx < fx){
                        x0 = x0 + rate * s_i;
                        fx = new_fx;
                    }else{
                        s_i *= -1;
                        rate = brent<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i), this->stop_,
                                this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        new_fx = problem.value(x0 + rate * s_i);
                        ++this->current_.evaluations;
                        this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                        if(new_fx < fx){
                            x0 = x0 + rate * s_i;
                            fx = new_fx;
                        }
                    }
                    break;
                }
                case line_search::IntervalHalving : {
                    rate = interval_halving<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i),
                            this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance()) ;
                    scalar new_fx = problem.value(x0 + rate * s_i);
                    ++this->current_.evaluations;
                    this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                    if(new_fx < fx){
                        x0 = x0 + rate * s_i;
                        fx = new_fx;
                    }else{
                        s_i *= -1;
                        rate = interval_halving<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i),
                                this->stop_, this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        new_fx = problem.value(x0 + rate * s_i);
                        ++this->current_.evaluations;
                        this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                        if(new_fx < fx){
                            x0 = x0 + rate * s_i;
                            fx = new_fx;
                        }
                    }
                    break;
                }
                case line_search::GoldenSection : {
                    rate = golden_section<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i), this->stop_,
                            this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance()) ;
                    scalar new_fx = problem.value(x0 + rate * s_i);
                    ++this->current_.evaluations;
                    this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                    if(new_fx < fx){
                        x0 = x0 + rate * s_i;
                        fx = new_fx;
                    }else{
                        s_i *= -1;
                        rate = golden_section<problem_type, 1>::linesearch(x0, s_i, problem, this->max_step_size_line_search(problem, x0, s_i), this->stop_,
                                this->current_, this->m_status, this->m_stats, this->options_.get_storage_criteria(), this->options_.get_line_search_tolerance());
                        new_fx = problem.value(x0 + rate * s_i);
                        ++this->current_.evaluations;
                        this->m_stats.push(new_fx, this->current_.evaluations, this->options_.get_storage_criteria());
                        if(new_fx < fx){
                            x0 = x0 + rate * s_i;
                            fx = new_fx;
                        }
                    }
                    break;
                }
                default:
                    std::cerr << "Invalid line search algorithm." << std::endl;
            }
        }

        void minimize(problem_type &problem, vector_t &x0) {

            this->current_.reset();
            const size_t dim = x0.size();

            matrix_type s = matrix_type::Identity(dim, dim);
            vector_t s_i = vector_t::Zero(dim);
            vector_t z = vector_t::Zero(dim);
            scalar fx;

            fx = problem.value(x0);
            ++this->current_.evaluations;
            this->m_stats.push(fx, this->current_.evaluations, this->options_.get_storage_criteria());

            s_i = s.row(dim-1);

            apply_line_search(x0, s_i, problem, fx);

            this->current_.fx_best = fx;

            do{
                z = x0;
                for(size_t i = 0; i < dim; i++){
                    s_i = s.row(i);
                    apply_line_search(x0, s_i, problem, fx);
                }

                s_i = x0 - z;
                apply_line_search(x0, s_i, problem, fx);

                // set new S
                for(size_t i = 0; i < dim - 1; i++){
                    for(size_t j = 0; j < dim; j++){
                        s(i, j) = s(i+1, j);
                    }
                }
                for(size_t j = 0; j < dim; j++){
                    s(dim-1, j) = s_i[j];
                }

                ++this->current_.iterations;
                this->current_.fx_best = fx;
                this->m_status = checkConvergence(this->stop_, this->current_);

                if(this->m_debug >= debug_level::Low) {
                    cout << "Iterations: " << this->current_.iterations << " - Evaluations: "
                         << this->current_.evaluations << " - f(x):" << this->current_.fx_best << endl;
                }

            }while(problem.callback(this->current_, x0) && (this->m_status == status::Continue));
        }
    };

}

#endif