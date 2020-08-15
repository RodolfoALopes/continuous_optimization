#ifndef CONTINUOUS_OPTIMIZATION_F7_BBO_CEC_2013_H
#define CONTINUOUS_OPTIMIZATION_F7_BBO_CEC_2013_H

#include <cec2013/F7.h>
#include <solver/problem/bounded_problem_interface.h>

using namespace solver;

template<typename T>
class f7_bbo_cec_2013 : public bounded_problem_interface<T> {
public:
    using Superclass = bounded_problem_interface<T>;
    using typename Superclass::vector_t;

protected:
    F7 f;
    T fx_star = 0.0;

public:
    explicit f7_bbo_cec_2013(int dim) : Superclass(dim) {
        this->setLowerBound(vector_t::Ones(dim) * f.getMinX());
        this->setUpperBound(vector_t::Ones(dim) * f.getMaxX());
    }

    T get_fx_start(){
        return fx_star;
    }

    T value(const vector_t &x) override {
        const size_t dim = x.size();
        double *x_ = new double[dim];
        for(size_t i = 0; i < dim; i++){
            x_[i] = x[i];
        }
        T fx = f.compute(x_);
        delete[] x_;
        return fx;
    }
};

#endif