#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_F14_GRAYBOX_CEC_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_F14_GRAYBOX_CEC_H

#include <solver/problem/bounded_problem_interface.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_math.h>
#include <solver/problem/bounded_problems/gbo_cec_2013_benchmark/cec_read_files.h>

using namespace solver;

template<typename T>
class f14_graybox_cec : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

protected:
    const size_t id_function = 14;
    T fx_star = 0.0;
    T min_bound = -100.0;
    T max_bound = 100.0;
    matrix_type o;
    vector_t w;
    vector<size_t> p;
    vector<size_t> s;
    matrix_type r25;
    matrix_type r50;
    matrix_type r100;
    const size_t overlap = 5;
    const size_t number_sub_component = 20;
//    const size_t dim_overlapping = 905; // because of overlapping


public:
    explicit f14_graybox_cec(int dim) : Superclass(dim) {
        cec_read_files read;
        o = matrix_type(number_sub_component, 100);
        w = vector_t(number_sub_component);
        p = vector<size_t>(dim, 0);
        s = vector<size_t>(number_sub_component, 0);
        r25 = matrix_type(25, 25);
        r50 = matrix_type(50, 50);
        r100 = matrix_type(100, 100);

        read.read_matrix_files(o, id_function, cec2013_type_file::CEC2013_SHIFT_MATRIX);
        read.read_vector_files(p, id_function, cec2013_type_file::CEC2013_PERM_VECTOR);
        read.read_vector_files(w, id_function, cec2013_type_file::CEC2013_WEIGHT_VECTOR);
        read.read_vector_files(s, id_function, cec2013_type_file::CEC2013_SUB_COMPONENTS_VECTOR);
        read.read_matrix_files(r25, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_25);
        read.read_matrix_files(r50, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_50);
        read.read_matrix_files(r100, id_function, cec2013_type_file::CEC2013_ROTATION_MATRIX_100);

        this->setLowerBound(vector_t::Ones(dim) * min_bound);
        this->setUpperBound(vector_t::Ones(dim) * max_bound);
        this->vig = variable_interaction_graph(dim);

        vector<size_t> x(dim);
        for(size_t i = 0; i < dim; i++){
            x[i] = i;
            this->variable_to_sub_function[i] = std::make_pair(true,0);
        }
        this->sub_function_to_variable.push_back(x);

        size_t global_index = 0;
        size_t source, dest;
        for(size_t i = 0; i < number_sub_component; i++){
            for(size_t j = global_index - i*overlap; j < global_index +s[i] - i*overlap; ++j){
                source = p[j];
                for(size_t k = global_index - i*overlap; k < global_index +s[i] - i*overlap; ++k){
                    dest = p[k];
                    this->vig.add_edge(source, dest);
                }
            }
            global_index += s[i];
        }
        this->vig.pre_processing_graph();
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        cec_math<T> math;
        T result = 0.0;
        size_t global_index = 0;
        for (size_t i = 0; i < number_sub_component; i++){
            size_t sub_component_size = s[i];
            vector_t z(sub_component_size);

            switch(sub_component_size){
                case 25:{
                    math.rotateVectorConflict(x, p, z, o, r25, sub_component_size, global_index, i, overlap);
                    break;
                }
                case 50:{
                    math.rotateVectorConflict(x, p, z, o, r50, sub_component_size, global_index, i, overlap);
                    break;
                }
                case 100:{
                    math.rotateVectorConflict(x, p, z, o, r100, sub_component_size, global_index, i, overlap);
                    break;
                }
                default:
                    throw std::logic_error("Invalid sub component size");
            }

            global_index += s[i];

            T r = math.schwefel(z);
            result += w[i] * r;
        }
        return result;
    }
};

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_F14_GRAYBOX_CEC_H
