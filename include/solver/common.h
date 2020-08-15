#ifndef CONTINUOUS_OPTIMIZATION_COMMON_H
#define CONTINUOUS_OPTIMIZATION_COMMON_H


#include <string>
#include <iostream>
#include <chrono>
#include <Eigen/Core>

namespace solver {

    enum class debug_level { None = 0, Low, High };

    enum class storage_criteria_level {All_Best_Solutions, CEC_Stop_Points};

    enum class status {
        NotStarted = -1,
        Continue = 0,
        IterationLimit,
        EvaluationLimit,
        XDeltaTolerance,
        FDeltaTolerance,
        GradNormTolerance,
        Condition,
        FOptimum,
        UserDefined
    };

    enum class simplex_op {
        Place,
        Reflect,
        Expand,
        ContractIn,
        ContractOut,
        Shrink
    };

    enum class line_search{
        AcceleratedStepSize,
        Brent,
        GoldenSection,
        IntervalHalving
    };

    template<typename T>
    T max(const T a, const T b){
        return (a > b ? a : b);
    }

    template<typename T>
    T min(const T a, const T b){
        return (a < b ? a : b);
    }

    template<typename T>
    T max_limits(){
        return (T)std::numeric_limits<double>::max();
    }

    template<typename T>
    class options {

    protected:

        // General Parameter
        bool use_decomposition_functions;
        storage_criteria_level storage_criteria;

        // Parameters for Line Search
        line_search line_search_;
        T line_search_tolerance_;
        T alpha_initial_;
        T max_bound_;

        // LSGSS parameters
        size_t number_groups;
        size_t number_steps;
        T beta;
        T alpha;
        bool auto_update_steps;

    public:

        static options defaults() {
            options op_;

            // General Parameters
            op_.use_decomposition_functions = false;
            op_.storage_criteria = storage_criteria_level::All_Best_Solutions;

            // Line Search - Parameters
            op_.line_search_ = line_search::AcceleratedStepSize;
            op_.line_search_tolerance_ = 1e-6;
            op_.alpha_initial_ = 0.0001;
            op_.max_bound_ = 1000.0;

            // LSGSS - Parameters
            op_.number_groups = 2;
            op_.number_steps = 2;
            op_.beta = 0.025;
            op_.alpha = 0.25;
            op_.auto_update_steps = true;

            return op_;
        }

        line_search get_line_search() {
            return line_search_;
        }

        void set_line_search(line_search l) {
            line_search_ = l;
        }

        T get_line_search_tolerance() {
            return line_search_tolerance_;
        }

        void set_line_search_tolerance(T l) {
            line_search_tolerance_ = l;
        }

        T get_alpha_initial() {
            return alpha_initial_;
        }

        void set_alpha_initial(T a) {
            alpha_initial_ = a;
        }

        T get_max_bound() {
            return max_bound_;
        }

        void set_max_bound(T m) {
            max_bound_ = m;
        }

        size_t get_number_groups() {
            return number_groups;
        }

        void set_number_groups(size_t l) {
            number_groups = l;
        }

        void set_number_steps(size_t i) {
            number_steps = i;
        }

        size_t get_number_steps() {
            return number_steps;
        }

        T get_beta() {
            return beta;
        }

        void set_beta(T d) {
            beta = d;
        }

        T get_alpha() {
            return alpha;
        }

        void set_alpha(T s) {
            alpha = s;
        }

        bool get_auto_update_steps() {
            return auto_update_steps;
        }

        void set_auto_update_steps(bool s) {
            auto_update_steps = s;
        }

        void set_use_decomposition_function(bool v) {
            use_decomposition_functions = v;
        }

        bool get_use_decomposition_function() {
            return use_decomposition_functions;
        }

        void set_storage_criteria(storage_criteria_level s) {
            storage_criteria = s;
        }

        storage_criteria_level get_storage_criteria() {
            return storage_criteria;
        }
    };

    inline std::ostream &operator<<(std::ostream &os, const simplex_op &op) {
        switch (op) {
            case simplex_op::Place: os << "place"; break;
            case simplex_op::Reflect:   os << "reflect"; break;
            case simplex_op::Expand:   os << "expand"; break;
            case simplex_op::ContractIn:   os << "contract-in"; break;
            case simplex_op::ContractOut:   os << "contract-out"; break;
            case simplex_op::Shrink:   os << "shrink"; break;
        }
        return os;
    }

    inline std::string op_to_string(simplex_op op) {
        switch (op) {
            case simplex_op::Place:
                return "place";
            case simplex_op::Expand:
                return "expand";
            case simplex_op::Reflect:
                return "reflect";
            case simplex_op::ContractIn:
                return "contract-in";
            case simplex_op::ContractOut:
                return "contract-out";
            case simplex_op::Shrink:
                return "shrink";
        }
        return "unknown";
    }

    template<typename T>
    class criteria {
    public:
        long iterations;     //!< Maximum number of iterations
        long evaluations;    //!< Maximum number of function evaluations
        T xDelta;            //!< Minimum change in parameter vector
        T fDelta;            //!< Minimum change in cost function
        T gradNorm;          //!< Minimum norm of gradient vector
        T condition;         //!< Maximum condition number of Hessian
        bool fx_is_know;     //!< If the f(x*) is know
        T fx_best;           //!< Value of f(x*)
        T error_fx_best;     // Acceptable error -> f(x*) - f(x) <= error

        criteria() {
            reset();
        }

        static criteria defaults() {
            criteria d;
            d.iterations = 10000;
            d.evaluations = 3e6;
            d.xDelta = 0;
            d.fDelta = 0;
            d.gradNorm = 1e-4;
            d.condition = 0;
            d.fx_is_know = false;
            d.fx_best = max_limits<T>();
            d.error_fx_best = 0.0;
            return d;
        }

        void reset() {
            iterations = 0;
            evaluations = 0;
            xDelta = 0;
            fDelta = 0;
            gradNorm = 0;
            condition = 0;
            fx_best = max_limits<T>();
        }

        void print(std::ostream &os) const {
            os << "Iterations:  " << iterations << std::endl;
            os << "Evaluations: " << evaluations << std::endl;
            os << "xDelta:      " << xDelta << std::endl;
            os << "fDelta:      " << fDelta << std::endl;
            os << "GradNorm:    " << gradNorm << std::endl;
            os << "Condition:   " << condition << std::endl;
            os << "f(x_best):   " << fx_best << std::endl;
        }
    };

    template<typename T>
    status checkConvergence(const criteria<T> &stop, const criteria<T> &current) {
        if ((stop.fx_is_know) && (current.fx_best - stop.fx_best <= stop.error_fx_best)) {
            return status::FOptimum;
        }
        if ((stop.iterations > 0) && (current.iterations >= stop.iterations)) {
            return status::IterationLimit;
        }
        if ((stop.evaluations > 0) && (current.evaluations >= stop.evaluations)) {
            return status::EvaluationLimit;
        }
        if ((stop.xDelta > 0) && (current.xDelta < stop.xDelta)) {
            return status::XDeltaTolerance;
        }
        if ((stop.fDelta > 0) && (current.fDelta < stop.fDelta)) {
            return status::FDeltaTolerance;
        }
        if ((stop.gradNorm > 0) && (current.gradNorm < stop.gradNorm)) {
            return status::GradNormTolerance;
        }
        if ((stop.condition > 0) && (current.condition > stop.condition)) {
            return status::Condition;
        }
        return status::Continue;
    }

    inline std::ostream &operator<<(std::ostream &os, const status &s) {
        switch (s) {
            case status::IterationLimit: os << "Iteration limit reached."; break;
            case status::EvaluationLimit: os << "Function evaluation limit reached."; break;
            case status::FOptimum:   os << "f(x*) criteria reached."; break;
            case status::XDeltaTolerance: os << "Change in parameter vector too small."; break;
            case status::FDeltaTolerance: os << "Change in cost function value too small."; break;
            case status::GradNormTolerance: os << "Gradient vector norm too small."; break;
            case status::Condition: os << "Condition of Hessian/Covariance matrix too large."; break;
            case status::UserDefined: os << "Stop condition defined in the callback."; break;
            case status::NotStarted: os << "Solver not started."; break;
            case status::Continue:   os << "Convergence criteria not reached."; break;
        }
        return os;
    }

    template <typename T>
    class stats{
        struct stat{
            T fx;
            size_t evaluation;
            std::chrono::duration<double> time;
        };

    private:
        std::vector<stat> history;
        std::chrono::steady_clock::time_point starting_time;

    public:
        void push(T fx, size_t e, storage_criteria_level op){
            if(op == storage_criteria_level::All_Best_Solutions) {
                if (history.empty()) {
                    stat s;
                    s.fx = fx;
                    s.evaluation = e;
                    s.time = std::chrono::steady_clock::now() - starting_time;
                    history.push_back(s);
                } else if (fx < history.back().fx) {
                    stat s;
                    s.fx = fx;
                    s.evaluation = e;
                    s.time = std::chrono::steady_clock::now() - starting_time;
                    history.push_back(s);
                }
            }
            else if(op == storage_criteria_level::CEC_Stop_Points){
                if(history.empty()){
                    // long stop_1 = 1.2e5;
                    stat s;
                    s.fx = fx;
                    s.evaluation = e;
                    s.time = std::chrono::steady_clock::now() - starting_time;
                    history.push_back(s);
                    // long stop_2 = 6e5;
                    history.push_back(s);
                    // long stop_3 = 3e6;
                    history.push_back(s);
                }
                else{
                    if(e <= 1.2e5 && fx < history[0].fx){
                        stat s;
                        s.fx = fx;
                        s.evaluation = e;
                        s.time = std::chrono::steady_clock::now() - starting_time;
                        history[0] = s;
                    }
                    if(e <= 6e5 && fx < history[1].fx){
                        stat s;
                        s.fx = fx;
                        s.evaluation = e;
                        s.time = std::chrono::steady_clock::now() - starting_time;
                        history[1] = s;
                    }
                    if(e <= 3e6 && fx < history[2].fx){
                        stat s;
                        s.fx = fx;
                        s.evaluation = e;
                        s.time = std::chrono::steady_clock::now() - starting_time;
                        history[2] = s;
                    }
                }
            }
        }

        std::vector<stat> get_history(){
            return history;
        }

        void reset(){
            history.clear();
            starting_time = std::chrono::steady_clock::now();
        }
    };

    template<typename T>
    std::ostream &operator<<(std::ostream &os, const criteria<T> &c) {
        c.print(os);
        return os;
    }

}
#endif