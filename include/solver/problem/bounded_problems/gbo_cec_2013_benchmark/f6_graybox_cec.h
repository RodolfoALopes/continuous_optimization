#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_F6_GRAYBOX_CEC_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_F6_GRAYBOX_CEC_H

#include <solver/problem/bounded_problem_interface.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_math.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_read_files.h>

using namespace solver;

template<typename T>
class f6_graybox_cec : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

protected:
    const size_t id_function = 6;
    T fx_star = 0.0;
    T min_bound = -32.0;
    T max_bound = 32.0;
    vector_t o;
    vector_t w;
    vector<size_t> p;
    vector<size_t> s;
    matrix_type r25;
    matrix_type r50;
    matrix_type r100;
    const size_t number_sub_component = 7;
    const size_t number_seps_var = 700;

public:
    explicit f6_graybox_cec(int dim) : Superclass(dim) {
        o = vector_t(dim);
        w = vector_t(number_sub_component);
        p = vector<size_t>(dim, 0);
        s = vector<size_t>(number_sub_component, 0);
        r25 = matrix_type(25, 25);
        r50 = matrix_type(50, 50);
        r100 = matrix_type(100, 100);

        cec_read_files read;
        read.read_vector_files(o, id_function, cec2013_type_file::CEC2013_SHIFT_VECTOR);
        read.read_vector_files(p, id_function, cec2013_type_file::CEC2013_PERM_VECTOR);
        read.read_vector_files(w, id_function, cec2013_type_file::CEC2013_WEIGHT_VECTOR);
        read.read_vector_files(s, id_function, cec2013_type_file::CEC2013_SUB_COMPONENTS_VECTOR);
        read.read_matrix_files(r25, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_25);
        read.read_matrix_files(r50, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_50);
        read.read_matrix_files(r100, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_100);

        this->setLowerBound(vector_t::Ones(dim) * min_bound);
        this->setUpperBound(vector_t::Ones(dim) * max_bound);
        this->vig = variable_interaction_graph(dim);

        size_t global_index = 0;
        for(size_t i = 0; i < number_sub_component; i++){
            vector<size_t> x(s[i]);
            for(size_t j = 0; j < s[i]; j++){
                x[j] = p[global_index + j];
                this->variable_to_sub_function[p[global_index + j]] = std::make_pair(true,i);
                for(size_t k = j + 1; k < s[i]; k++) {
                    this->vig.add_edge(p[global_index + j], p[global_index + k]);
                }
            }
            this->sub_function_to_variable.push_back(x);
            global_index += s[i];
        }

        vector<size_t> x;
        x.reserve(dim - global_index);
        int k = number_sub_component;
        for(int i = dim - 1; i >= global_index; i--){
            this->variable_to_sub_function[p[i]] = std::make_pair(false, k);
            x.push_back(p[i]);
        }
        this->sub_function_to_variable.push_back(x);
        this->vig.pre_processing_graph();
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        const size_t dim = x.size();
        cec_math<T> math;
        T result = 0.0;
        size_t c = 0;
        size_t global_index = 0;
        vector_t z = x - o;
        for(size_t i = 0; i < number_sub_component; i++){
            size_t sub_component_size = s[i];
            c += sub_component_size;
            vector_t z_1(sub_component_size);
            switch(sub_component_size){
                case 25:{
                    math.rotateVector(z, p, z_1, r25, sub_component_size, global_index);
                    break;
                }
                case 50:{
                    math.rotateVector(z, p, z_1, r50, sub_component_size, global_index);
                    break;
                }
                case 100:{
                    math.rotateVector(z, p, z_1, r100, sub_component_size, global_index);
                    break;
                }
                default:
                    throw std::logic_error("Invalid sub component size");
            }
            global_index += s[i];
            T r = math.ackley(z_1);
            result += w[i] * r;
        }
        vector_t z_1(dim - c);
        for (size_t i = c; i < dim; i++){
            z_1[i-c] = z[p[i]];
        }
        T r = math.ackley(z_1);
        result += r;
        return result;
    }

    T delta(const vector_t &x, T fx, T delta_i, size_t i) override {
        vector_t x_delta = x;
        x_delta[i] = delta_i;
        const size_t dim = x.size();
        cec_math<T> math;
        size_t global_index = 0;
        T result = 0.0;
        vector_t z = x_delta - o;
        auto id_sub_function = this->variable_to_sub_function[i];
        if(id_sub_function.second < number_sub_component){
            for(size_t j = 0; j < id_sub_function.second; j++){
                global_index += s[j];
            }
            size_t sub_component_size = s[id_sub_function.second];
            vector_t z_1(sub_component_size);
            switch(sub_component_size){
                case 25:{
                    math.rotateVector(z, p, z_1, r25, sub_component_size, global_index);
                    break;
                }
                case 50:{
                    math.rotateVector(z, p, z_1, r50, sub_component_size, global_index);
                    break;
                }
                case 100:{
                    math.rotateVector(z, p, z_1, r100, sub_component_size, global_index);
                    break;
                }
                default:
                    throw std::logic_error("Invalid sub component size");
            }
            T r = math.ackley(z_1);
            result += w[id_sub_function.second] * r;
        }
        else{
            global_index = dim - number_seps_var;
            vector_t z_1(dim - global_index);
            for (size_t j = global_index; j < dim; j++){
                z_1[j-global_index] = z[p[j]];
            }
            result = math.ackley(z_1);
        }
        return result - fx;
    }

};

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_F6_GRAYBOX_CEC_H
