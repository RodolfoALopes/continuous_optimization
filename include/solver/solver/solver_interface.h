#ifndef CONTINUOUS_OPTIMIZATION_SOLVER_H_
#define CONTINUOUS_OPTIMIZATION_SOLVER_H_

#include <functional>
#include <solver/common.h>
#include <solver/problem/problem_interface.h>
#include <solver/problem/bounded_problem_interface.h>

namespace solver {

    template<typename problem_type, int Ord>
    class solver_interface {
    public:
        using scalar     = typename problem_type::scalar;
        using vector_t   = typename problem_type::vector_t;
        using hessian_t  = typename problem_type::hessian_t;
        using criteria_t = typename problem_type::criteria_t;
    protected:
        const int order_ = Ord;
        criteria_t stop_, current_;
        options<scalar> options_;
        status m_status = status::NotStarted;
        stats<scalar> m_stats;
        debug_level m_debug = debug_level::None;

    public:
        virtual ~solver_interface() = default;
        solver_interface() {
            stop_ = criteria_t::defaults();
            options_ = options<scalar>::defaults();
            current_.reset();
            m_stats.reset();
        }

        solver_interface(const criteria_t &s) {
            stop_ = s;
            current_.reset();
            m_stats.reset();
            options_ = options<scalar>::defaults();
        }

        void set_stop_criteria(const criteria_t &s) { stop_ = s; }

        void set_options(const options<scalar> &op) { options_ = op; }

        const criteria_t &criteria() { return current_; }

        const status &get_status() { return m_status; }

        stats<scalar> &get_stats() { return m_stats; }

        void set_debug(const debug_level &d) { m_debug = d; }

        virtual void minimize(problem_type &objective_function, vector_t &x0) = 0;

        template <class T = problem_type, class SCALAR = typename problem_type::scalar, int N = problem_type::Dim>
        std::enable_if_t<std::is_base_of_v<bounded_problem_interface<SCALAR,N>,T>,double>
        max_step_size_line_search(const problem_type &problem, const vector_t &x, const vector_t &search_direction) {
            scalar largest_double_without_breaking_bounds = max_limits<scalar>();
            const scalar dim = search_direction.size();
            for(size_t i = 0; i < dim; i++){
                const scalar value = x[i];
                const scalar direction = search_direction[i];
                if(direction != 0.0){
                    const bool direction_is_increasing = direction > 0.0;
                    const scalar bound = direction_is_increasing ? problem.upperBound()[i] : problem.lowerBound()[i];
                    largest_double_without_breaking_bounds = min(largest_double_without_breaking_bounds, (bound - value)/direction);
                }
            }
            if(largest_double_without_breaking_bounds == max_limits<scalar>())
                return 0.0;
            else
                return largest_double_without_breaking_bounds;
        }

        template <class T = problem_type, class SCALAR = typename problem_type::scalar, int N = problem_type::Dim>
        std::enable_if_t<!std::is_base_of_v<bounded_problem_interface<SCALAR,N>,T>,double>
        max_step_size_line_search(const problem_type &problem, const vector_t &x, const vector_t &search_direction) {
            return this->options_.get_max_bound();
        }

    };
}

#endif