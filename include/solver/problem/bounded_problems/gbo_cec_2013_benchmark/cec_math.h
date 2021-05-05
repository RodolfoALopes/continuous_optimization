#ifndef CONTINUOUS_NON_LINEAR_OPTIMIZATION_CEC_MATH_H
#define CONTINUOUS_NON_LINEAR_OPTIMIZATION_CEC_MATH_H

namespace solver {

    template<typename scalar>
    class cec_math{

    protected:
        scalar const beta = 0.2;
        scalar const alpha = 10.0;
        scalar const PI = 3.141592653589793238462643383279;
        scalar const E = 2.718281828459045235360287471352;

        int sign(scalar x){
            if (x > 0) return 1;
            if (x < 0) return -1;
            return 0;
        }

        scalar hat(scalar x){
            return (x == 0 ? 0.0 : log(abs(x)));
        }

        scalar c1(scalar x){
            return (x > 0 ? 10.0 : 5.5);
        }

        scalar c2(scalar x){
            return (x > 0 ? 7.9 : 3.1);
        }

    public:
        cec_math() = default;

        void transform_osz_value(scalar &x){
            x = sign(x) * exp(hat(x) + 0.049 * (sin(c1(x) * hat(x)) + sin(c2(x)* hat(x))));
        }

        template<typename vector_real>
        void transform_osz_vector(vector_real &x){
            const size_t dim = x.size();
            for(size_t i = 0; i < dim; ++i){
                transform_osz_value(x[i]);
            }
        }

        void transform_asy_value(scalar &x, size_t i, size_t dim){
            if (x > 0){
                x = pow(x, 1 + beta * i/((scalar) (dim-1)) * sqrt(x));
            }
        }

        void transform_asy_value(scalar &x, size_t i, size_t j, size_t dim){
            if (x>0){
                x = pow(x, 1 + beta * j/((scalar) (dim-1)) * sqrt(x));
            }
        }

        template<typename vector_real>
        void transform_asy_vector(vector_real &x){
            const size_t dim = x.size();
            for(size_t i = 0; i < dim; ++i){
                transform_asy_value(x[i], i, dim);
            }
        }

        void lambda_value(scalar &x, size_t i, size_t dim){
            x = x * pow(alpha, 0.5 * i/((scalar) (dim-1)));
        }

        void lambda_value(scalar &x, size_t i, size_t j, size_t dim){
            x = x * pow(alpha, 0.5 * j/((scalar) (dim-1)));
        }

        template<typename vector_real>
        void lambda_vector(vector_real &x){
            const size_t dim = x.size();
            for (int i = 0; i < dim; ++i){
                lambda_value(x[i], i, dim);
            }
        }

        template<typename vector_real, typename matrix_type>
        void multiply(vector_real &z_1, const matrix_type &matrix_r){
            const size_t dim = z_1.size();
            vector_real result(dim);
            for(int i = dim - 1; i >= 0; i--){
                result[i] = 0;
                for(int j = dim - 1; j >= 0; j--){
                    result[i] += z_1[j] * matrix_r(i,j);
                }
            }
            for(size_t i = 0; i < dim; i++){
                z_1[i] = result[i];
            }
        }

        template<typename vector_real, typename vector_integer, typename matrix_type>
        void rotateVector(const vector_real &z, const vector_integer &p, vector_real &z_1, const matrix_type &matrix_r,
                size_t sub_component_size, size_t global_index){
            for (int j = global_index; j < global_index + sub_component_size; ++j){
                z_1[j - global_index] = z[p[j]];
            }
            multiply(z_1, matrix_r);
        }

        template<typename vector_real, typename vector_integer, typename matrix_type>
        void rotateVectorConform(const vector_real &z, const vector_integer &p, vector_real &z_1, const matrix_type &matrix_r,
                size_t sub_component_size, size_t global_index, size_t i_sub_component, size_t overlap){
            for (size_t j = global_index - i_sub_component * overlap; j < global_index + sub_component_size - i_sub_component * overlap; ++j){
                z_1[j - (global_index - i_sub_component * overlap)] = z[p[j]];
            }
            multiply(z_1, matrix_r);
        }

        template<typename vector_real, typename vector_integer, typename matrix_type>
        void rotateVectorConflict(const vector_real &x, const vector_integer &p, vector_real &z, const matrix_type &o,
                                    const matrix_type &matrix_r, size_t sub_component_size, size_t global_index,
                                    size_t i_sub_component, size_t overlap){
            for (size_t j = global_index - i_sub_component * overlap; j < global_index + sub_component_size - i_sub_component * overlap; ++j){
                z[j - (global_index - i_sub_component * overlap)] = x[p[j]] - o(i_sub_component, (j- (global_index - i_sub_component * overlap)));
            }
            multiply(z, matrix_r);
        }

        template<typename vector_real>
        scalar rosenbrock(vector_real &x){
            const size_t dim = x.size();
            scalar result = 0.0;
            scalar oz, y;
            int i = dim - 1;
            for (--i; i >= 0; i--) {
                oz = x[i + 1];
                y  = ((x[i] * x[i]) - oz);
                result += (100.0 * y * y);
                y  = (x[i] - 1.0);
                result += (y * y);
            }
            return result;
        }

        scalar sphere_value(scalar x){
            scalar result = pow(x, 2);
            return result;
        }

        template<typename vector_real, typename vector_integer>
        scalar sphere(vector_real &x, const vector_integer &p, size_t begin_index){
            const size_t dim = x.size();
            scalar result = 0;
            for(int j = dim - 1; j >= begin_index; j--){
                result += sphere_value(x[p[j]]);
            }
            return result;
        }

        template<typename vector_real>
        scalar schwefel(vector_real &x){
            const size_t dim = x.size();
            scalar s1 = 0;
            scalar result = 0;
            transform_osz_vector(x);
            transform_asy_vector(x);
            for(size_t i = 0; i < dim; i++) {
                s1 += x[i];
                result += (s1 * s1);
            }
            return result;
        }


        template<typename vector_real>
        scalar ackley(vector_real &x){
            const size_t dim = x.size();
            scalar sum1 = 0.0;
            scalar sum2 = 0.0;
            scalar result;
            transform_osz_vector(x);
            transform_asy_vector(x);
            lambda_vector(x);
            for(int i = dim - 1; i >= 0; i--) {
                sum1 += (x[i] * x[i]);
                sum2 += cos(2.0 * PI * x[i]);
            }
            result = -20.0 * exp(-0.2 * sqrt(sum1 / dim)) - exp(sum2 / dim) + 20.0 + E;
            return result;
        }

        template<typename vector_t>
        scalar rastrigin(vector_t &x){
            const size_t dim = x.size();
            scalar result = 0.0;
            transform_osz_vector(x);
            transform_asy_vector(x);
            lambda_vector(x);
            for(int i = dim - 1; i >= 0; i--) {
                result += x[i] * x[i] - 10.0 * cos(2 * PI * x[i]) + 10.0;
            }
            return result;
        }

        scalar rastrigin(scalar x, size_t i, size_t dim){
            transform_osz_value(x);
            transform_asy_value(x, i, dim);
            lambda_value(x, i, dim);
            scalar result = x * x - 10.0 * cos(2 * PI * x) + 10.0;
            return result;
        }

        template<typename vector_real, typename vector_integer>
        scalar rastrigin(vector_real &x, const vector_integer &p, size_t i, size_t j, size_t dim){
            transform_osz_value(x[p[i]]);
            transform_asy_value(x[p[i]], i, j, dim);
            lambda_value(x[p[i]], i, j, dim);
            scalar result = x[p[i]] * x[p[i]] - 10.0 * cos(2 * PI * x[p[i]]) + 10.0;
            return result;
        }

        template<typename vector_t>
        scalar elliptic(vector_t &x){
            const size_t dim = x.size();
            scalar result = 0.0;
            transform_osz_vector(x);
            for(size_t i = 0; i < dim; i++){
                scalar r = pow(1.0e6,  i/((scalar)(dim - 1)) ) * x[i] * x[i];
                result += r;
            }
            return result;
        }

        template<typename vector_t>
        scalar elliptic(vector_t &x, size_t i){
            const size_t dim = x.size();
            scalar result = 0.0;
            transform_osz_value(x[i]);
            result = pow(1.0e6,  i/((scalar)(dim - 1)) ) * x[i] * x[i];
            return result;
        }

        template<typename vector_t>
        scalar elliptic(vector_t &x, const vector<size_t> &p, size_t i, size_t j, size_t dim){
            transform_osz_value(x[p[i]]);
            scalar result = pow(1.0e6,  j/((scalar)(dim - 1)) ) * x[(int)p[i]] * x[(int)p[i]];
            return result;
        }
    };
}

#endif //CONTINUOUS_NON_LINEAR_OPTIMIZATION_CEC_MATH_H
