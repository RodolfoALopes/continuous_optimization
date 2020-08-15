#ifndef BOUNDED_PROBLEM_H
#define BOUNDED_PROBLEM_H

#include <vector>
#include <Eigen/Core>

#include "problem_interface.h"

namespace solver {

    template<typename scalar_, int CompileDim_ = Eigen::Dynamic>
    class bounded_problem_interface : public problem_interface<scalar_, CompileDim_> {
    public:
        using superclass = problem_interface<scalar_, CompileDim_>;
        using typename superclass::scalar;
        using typename superclass::vector_t;

    protected:
        vector_t m_lowerBound;
        vector_t m_upperBound;

    public:
        bounded_problem_interface(int RunDim = CompileDim_) : superclass() {
            vector_t infBound(RunDim);
            infBound.setConstant(max_limits<scalar>());
            m_lowerBound = -infBound;
            m_upperBound = infBound;
        }

        bounded_problem_interface(const vector_t &l, const vector_t &u) :
            superclass(),
            m_lowerBound(l),
            m_upperBound(u)
        {}

        const vector_t &lowerBound() const { return m_lowerBound; }
        void setLowerBound(const vector_t &lb) { m_lowerBound = lb; }
        const vector_t &upperBound() const { return m_upperBound; }
        void setUpperBound(const vector_t &ub) { m_upperBound = ub; }

        void setBoxConstraint(vector_t  lb, vector_t  ub) {
            setLowerBound(lb);
            setUpperBound(ub);
        }

        bool isValid(const vector_t &x){
            const scalar err = 0.000000000001;
            for(size_t i = 0; i < x.size(); i++){
                if((x[i] > m_upperBound[i] + err) || (x[i] < m_lowerBound[i] - err))
                    return false;
            }
            return true;
        }
    };

}
#endif