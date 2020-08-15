#ifndef PROBLEM_H
#define PROBLEM_H

#include <array>
#include <vector>
#include <unordered_map>
#include <Eigen/Core>
#include <solver/common.h>
#include <solver/variable_interaction_graph.h>

namespace solver {

    template<typename scalar_, int Dim_ = Eigen::Dynamic>
    class problem_interface {
    public:
        static const int Dim = Dim_;
        typedef scalar_ scalar;
        using vector_t   = Eigen::Matrix<scalar, Dim, 1>;
        using hessian_t  = Eigen::Matrix<scalar, Dim, Dim>;
        using criteria_t = criteria<scalar>;
        using TIndex = typename vector_t::Index;
        using matrix_type = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;

    protected:
        variable_interaction_graph vig;
        unordered_map<size_t, std::pair<bool, size_t>> variable_to_sub_function;
        vector<vector<size_t>> sub_function_to_variable;
        vector<vector<size_t>> sub_problem_to_variable;

    public:
        problem_interface() {
            vig = variable_interaction_graph();
        }
        virtual ~problem_interface()= default;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
        virtual bool callback(const criteria<scalar> &state, const vector_t &x) {
            return true;
        }

        virtual bool detailed_callback(const criteria<scalar> &state, simplex_op op, int index, const matrix_type &x, std::vector<scalar> f) {
            return true;
        }
#pragma GCC diagnostic pop

        virtual scalar value(const  vector_t &x) = 0;

        virtual scalar value(const  vector_t &x, bool imprimir){
            return 0.0;
        }

        scalar operator()(const  vector_t &x) {
            return value(x);
        }

        virtual void gradient(const  vector_t &x,  vector_t &grad) {
            finite_gradient(x, grad);
        }

        virtual void gradient(const  vector_t &x,  vector_t &grad, size_t i) {
            finite_gradient(x, grad, i);
        }

        virtual void hessian(const vector_t &x, hessian_t &hessian) {
            finite_hessian(x, hessian);
        }

        virtual bool check_gradient(const vector_t &x, int accuracy = 3) {
            const TIndex D = x.rows();
            vector_t actual_grad(D);
            vector_t expected_grad(D);
            gradient(x, actual_grad);
            finite_gradient(x, expected_grad, accuracy);
            for (TIndex d = 0; d < D; ++d) {
                scalar scale = max(max(fabs(actual_grad[d]), fabs(expected_grad[d])), scalar(1.));
                if(fabs(actual_grad[d]-expected_grad[d])>1e-2 * scale)
                    return false;
            }
            return true;
        }

        virtual bool check_hessian(const vector_t &x, int accuracy = 3) {
            const TIndex D = x.rows();
            hessian_t actual_hessian = hessian_t::Zero(D, D);
            hessian_t expected_hessian = hessian_t::Zero(D, D);
            hessian(x, actual_hessian);
            finite_hessian(x, expected_hessian, accuracy);
            for (TIndex d = 0; d < D; ++d) {
                for (TIndex e = 0; e < D; ++e) {
                    scalar scale = max(static_cast<scalar>(max(fabs(actual_hessian(d, e)), fabs(expected_hessian(d, e)))), scalar(1.));
                    if(fabs(actual_hessian(d, e)- expected_hessian(d, e))>1e-1 * scale)
                        return false;
                }
            }
            return true;
        }

        virtual scalar delta(const vector_t &x, scalar fx, scalar delta_i, size_t i){
            vector_t x_delta = x;
            x_delta[i] = delta_i;
            return value(x_delta) - fx;
        }

        virtual scalar delta(const vector_t &x, scalar fx, scalar delta_i, size_t i, bool imprimir){
            return 0.0;
        }

        void finite_gradient(const  vector_t &x, vector_t &grad, int accuracy = 0) {
            // accuracy can be 0, 1, 2, 3
            const scalar eps = 2.2204e-6;
            static const std::array<std::vector<scalar>, 4> coeff =
                    { { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} } };
            static const std::array<std::vector<scalar>, 4> coeff2 =
                    { { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} } };
            static const std::array<scalar, 4> dd = {2, 12, 60, 840};

            grad.resize(x.rows());
            vector_t& xx = const_cast<vector_t&>(x);

            const int innerSteps = 2*(accuracy+1);
            const scalar ddVal = dd[accuracy]*eps;

            for (TIndex d = 0; d < x.rows(); d++) {
                grad[d] = 0;
                for (int s = 0; s < innerSteps; ++s)
                {
                    scalar tmp = xx[d];
                    xx[d] += coeff2[accuracy][s]*eps;
                    grad[d] += coeff[accuracy][s]*value(xx);
                    xx[d] = tmp;
                }
                grad[d] /= ddVal;
            }
        }

        void finite_gradient(const  vector_t &x, vector_t &grad, size_t i, int accuracy = 0) {
            // accuracy can be 0, 1, 2, 3
            const scalar eps = 2.2204e-6;
            static const std::array<std::vector<scalar>, 4> coeff =
                    { { {1, -1}, {1, -8, 8, -1}, {-1, 9, -45, 45, -9, 1}, {3, -32, 168, -672, 672, -168, 32, -3} } };
            static const std::array<std::vector<scalar>, 4> coeff2 =
                    { { {1, -1}, {-2, -1, 1, 2}, {-3, -2, -1, 1, 2, 3}, {-4, -3, -2, -1, 1, 2, 3, 4} } };
            static const std::array<scalar, 4> dd = {2, 12, 60, 840};

            grad.resize(x.rows());
            vector_t& xx = const_cast<vector_t&>(x);

            const int innerSteps = 2*(accuracy+1);
            const scalar ddVal = dd[accuracy]*eps;

            grad[i] = 0;
            for (int s = 0; s < innerSteps; ++s) {
                scalar tmp = xx[i];
                xx[i] += coeff2[accuracy][s]*eps;
                grad[i] += coeff[accuracy][s]*value(xx);
                xx[i] = tmp;
            }
            grad[i] /= ddVal;
        }

        void finite_hessian(const vector_t &x, hessian_t &hessian, int accuracy = 0) {
            const scalar eps = std::numeric_limits<double>::epsilon()*10e7;

            hessian.resize(x.rows(), x.rows());
            vector_t& xx = const_cast<vector_t&>(x);

            if(accuracy == 0) {
                for (TIndex i = 0; i < x.rows(); i++) {
                    for (TIndex j = 0; j < x.rows(); j++) {
                        scalar tmpi = xx[i];
                        scalar tmpj = xx[j];

                        scalar f4 = value(xx);
                        xx[i] += eps;
                        xx[j] += eps;
                        scalar f1 = value(xx);
                        xx[j] -= eps;
                        scalar f2 = value(xx);
                        xx[j] += eps;
                        xx[i] -= eps;
                        scalar f3 = value(xx);
                        hessian(i, j) = (f1 - f2 - f3 + f4) / (eps * eps);

                        xx[i] = tmpi;
                        xx[j] = tmpj;
                    }
                }
            } else {
                /*
                  \displaystyle{{\frac{\partial^2{f}}{\partial{x}\partial{y}}}\approx
                  \frac{1}{600\,h^2} \left[\begin{matrix}
                    -63(f_{1,-2}+f_{2,-1}+f_{-2,1}+f_{-1,2})+\\
                    63(f_{-1,-2}+f_{-2,-1}+f_{1,2}+f_{2,1})+\\
                    44(f_{2,-2}+f_{-2,2}-f_{-2,-2}-f_{2,2})+\\
                    74(f_{-1,-1}+f_{1,1}-f_{1,-1}-f_{-1,1})
                  \end{matrix}\right] }
                */
                for (TIndex i = 0; i < x.rows(); i++) {
                    for (TIndex j = 0; j < x.rows(); j++) {
                        scalar tmpi = xx[i];
                        scalar tmpj = xx[j];

                        scalar term_1 = 0;
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += -2*eps;  term_1 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += -1*eps;  term_1 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += 1*eps;   term_1 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += 2*eps;   term_1 += value(xx);

                        scalar term_2 = 0;
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += -2*eps;  term_2 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += -1*eps;  term_2 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += 2*eps;   term_2 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += 1*eps;   term_2 += value(xx);

                        scalar term_3 = 0;
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += -2*eps;  term_3 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += 2*eps;   term_3 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -2*eps; xx[j] += -2*eps;  term_3 -= value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 2*eps;  xx[j] += 2*eps;   term_3 -= value(xx);

                        scalar term_4 = 0;
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += -1*eps;  term_4 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += 1*eps;   term_4 += value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += 1*eps;  xx[j] += -1*eps;  term_4 -= value(xx);
                        xx[i] = tmpi; xx[j] = tmpj; xx[i] += -1*eps; xx[j] += 1*eps;   term_4 -= value(xx);

                        xx[i] = tmpi;
                        xx[j] = tmpj;

                        hessian(i, j) = (-63 * term_1+63 * term_2+44 * term_3+74 * term_4)/(600.0 * eps * eps);
                    }
                }
            }

        }

        variable_interaction_graph get_vig(){
            return vig;
        }

        unordered_map<size_t, std::pair<bool, size_t>> get_variable_to_sub_function(){
            return variable_to_sub_function;
        }

        vector<vector<size_t>> get_sub_function_to_variable(){
            return sub_function_to_variable;
        }

        vector<vector<size_t>> get_sub_problem_to_variable(){
            return sub_problem_to_variable;
        }
    };
}

#endif