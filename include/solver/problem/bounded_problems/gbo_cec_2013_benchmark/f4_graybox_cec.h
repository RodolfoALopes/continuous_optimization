//
// Created by rodolfo on 13/03/2020.
//

#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_F4_GRAYBOX_CEC_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_F4_GRAYBOX_CEC_H

#include <solver/problem/bounded_problem_interface.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_math.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_read_files.h>

using namespace solver;

template<typename T>
class f4_graybox_cec : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

protected:
    const size_t id_function = 4;
    T fx_star = 0.0;
    T min_bound = -100.0;
    T max_bound = 100.0;
    vector_t o;
    vector_t w;
    vector<size_t> p;
    vector<size_t> s;
    vector<size_t> seps;
    matrix_type r25;
    matrix_type r50;
    matrix_type r100;
    const size_t number_sub_component = 7;
    const size_t number_seps_var = 700;

public:
    explicit f4_graybox_cec(int dim) : Superclass(dim) {
        o = vector_t(dim);
        w = vector_t(number_sub_component);
        p = vector<size_t>(dim, 0);
        seps = vector<size_t>(dim, 0);
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
                this->variable_to_sub_function[p[global_index + j]] = std::make_pair(true, i);
                for(size_t k = j + 1; k < s[i]; k++) {
                    this->vig.add_edge(p[global_index + j], p[global_index + k]);
                }
            }
            this->sub_function_to_variable.push_back(x);
            global_index += s[i];
        }

        size_t k = number_sub_component;
        for(size_t i = global_index; i < dim; i++){
            this->variable_to_sub_function[p[i]] = std::make_pair(false, k);
            vector<size_t> x(1);
            x[0] = p[i];
            this->sub_function_to_variable.push_back(x);
            seps[k] = i;
            k++;
        }
        this->vig.pre_processing_graph();
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        const size_t dim = x.size();
        cec_math<T> math;
        size_t global_index = 0;
        T result = 0.0;
        vector_t z = x - o;
        for(size_t i = 0; i < number_sub_component; i++){
            size_t sub_component_size = s[i];
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
            T r = math.elliptic(z_1);
            result += w[i] * r;
        }
        vector_t z_1(dim-global_index);
        for (size_t i = global_index; i < dim; i++){
            z_1[i-global_index] = z[p[i]];
            result += math.elliptic(z_1, i-global_index);
        }
        return result;
    }

    T delta(const vector_t &x, T fx, T delta_i, size_t i) override {
        const size_t dim = x.size();
        cec_math<T> math;
        vector_t x_delta = x;
        x_delta[i] = delta_i;
        size_t global_index = 0;
        T result = 0.0;
        vector_t z = x_delta - o;
        auto id_sub_function = this->variable_to_sub_function[i];
        if(id_sub_function.first){
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
            T r = math.elliptic(z_1);
            result += w[id_sub_function.second] * r;
        }
        else{
            global_index = dim - number_seps_var;
            size_t index = seps[id_sub_function.second];
            result = math.elliptic(z, p, index, (index - global_index), (dim - global_index));
        }
        return result - fx;
    }

};

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_F4_GRAYBOX_CEC_H
