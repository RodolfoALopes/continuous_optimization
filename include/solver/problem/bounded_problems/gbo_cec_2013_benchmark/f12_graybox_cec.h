#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_F12_GRAYBOX_CEC_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_F12_GRAYBOX_CEC_H

#include <solver/problem/bounded_problem_interface.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_math.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_read_files.h>

using namespace solver;

template<typename T>
class f12_graybox_cec : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

protected:
    const size_t id_function = 12;
    T fx_star = 0.0;
    T min_bound = -100.0;
    T max_bound = 100.0;
    vector_t o;
    const size_t number_sub_component = 1;

public:
    explicit f12_graybox_cec(int dim) : Superclass(dim) {
        o = vector_t(dim);

        cec_read_files read;
        read.read_vector_files(o, id_function, cec2013_type_file::CEC2013_SHIFT_VECTOR);

        this->setLowerBound(vector_t::Ones(dim) * min_bound);
        this->setUpperBound(vector_t::Ones(dim) * max_bound);
        this->vig = variable_interaction_graph(dim);

        vector<size_t> x(dim);
        for(size_t i = 0; i < dim-1; i++){
            x[i] = i;
            this->variable_to_sub_function[i] = std::make_pair(true,0);
            this->vig.add_edge(i, i+1);
        }
        x[dim-1] = dim-1;
        this->sub_function_to_variable.push_back(x);
        this->vig.pre_processing_graph();
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        cec_math<T> math;
        vector_t z = x - o;
        T result = math.rosenbrock(z);
        return result;
    }
};

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_F12_GRAYBOX_CEC_H
