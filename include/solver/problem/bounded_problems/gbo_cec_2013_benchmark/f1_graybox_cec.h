#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_F1_GRAYBOX_CEC_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_F1_GRAYBOX_CEC_H

#include <solver/problem/bounded_problem_interface.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_math.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_read_files.h>

using namespace solver;

template<typename T>
class f1_graybox_cec : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;

protected:
    const size_t id_function = 1;
    T fx_star = 0.0;
    T min_bound = -100.0;
    T max_bound = 100.0;
    vector_t o;

public:
    explicit f1_graybox_cec(int dim) : Superclass(dim) {
        o = vector_t(dim);
        cec_read_files read;
        read.read_vector_files(o, id_function, cec2013_type_file::CEC2013_SHIFT_VECTOR);
        this->setLowerBound(vector_t::Ones(dim) * min_bound);
        this->setUpperBound(vector_t::Ones(dim) * max_bound);
        this->vig = variable_interaction_graph(dim);
        for(size_t i = 0; i < dim; i++){
            this->variable_to_sub_function[i] = std::make_pair(false, i);
            vector<size_t> x(1);
            x[0] = i;
            this->sub_function_to_variable.push_back(x);
        }
        this->vig.pre_processing_graph();
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        const size_t dim = x.size();
        cec_math<T> math;
        vector_t z = x - o;
        T result = 0.0;
        for(size_t i = 0; i < dim; i++){
            result += math.elliptic(z, i);
        }
        return result;
    }

    T delta(const vector_t &x, T fx, T delta_i, size_t i) override {
        vector_t x_delta = x;
        x_delta[i] = delta_i;
        cec_math<T> math;
        x_delta[i] = x_delta[i] - o[i];
        T result = math.elliptic(x_delta, i);
        return result - fx;
    }

};

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_F1_GRAYBOX_CEC_H
