#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>
#include <utility>

template <typename Tensor>
concept EigenTensor = std::is_base_of_v<Eigen::TensorBase<Tensor>, Tensor>;

namespace optim {

template <typename Derived>
class Optimizer {
public:
    virtual ~Optimizer() = default;
    auto Step(auto &params, const auto &grads) -> void {
        static_cast<Derived *>(this)->Step(params, grads);
    }
};

template <typename Expr>
auto Eval(const Expr &expr) {
    return Eigen::Tensor<typename Expr::Scalar, Expr::NumDimensions>(expr);
}

template <typename T>
constexpr auto getSize(const T &params) -> decltype(auto) {
    return std::tuple_size<typename std::remove_const<
        typename std::remove_reference<decltype(params)>::type>::type>::value;
}

class SGD : public Optimizer<SGD> {
private:
    template <auto N>
    constexpr auto InitVelocity(auto &grads) -> void {
        [&grads, this]<auto... Is>(std::index_sequence<Is...>) {
            velocity_ =
                std::make_tuple((Eval(std::get<Is>(grads).constant(0)))...);
        }(std::make_index_sequence<N>());
    }
    template <auto N>
    auto SGDImpl(auto &params, auto &grads) -> void {
        [&params, &grads, this]<auto... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(params) -= learning_rate_ * std::get<Is>(grads)),
             ...);
        }(std::make_index_sequence<N>());
    }
    template <auto N>
    auto MomentumImpl(auto &params, auto &grads) -> void {
        [&params, &grads, this]<auto... Is>(std::index_sequence<Is...>) {
            auto &velocity = std::any_cast<decltype(std::make_tuple(
                (Eval(std::get<Is>(grads).constant(0)))...)) &>(velocity_);
            ((std::get<Is>(velocity) =
                  momentum_ * std::get<Is>(velocity) + std::get<Is>(grads),
              std::get<Is>(params) -= learning_rate_ * std::get<Is>(velocity)),
             ...);
        }(std::make_index_sequence<N>());
    }
    float learning_rate_, momentum_;
    std::any velocity_;

public:
    SGD(float learning_rate = 1e-3, float momentum = 0)
        : learning_rate_(learning_rate), momentum_(momentum) {}
    auto Step(auto &params, auto &grads) -> void {
        constexpr auto size = getSize(params);
        if (momentum_) {
            if (!velocity_.has_value()) {
                InitVelocity<size>(grads);
            }
            MomentumImpl<size>(params, grads);
        } else {
            SGDImpl<size>(params, grads);
        }
    }
};

class Adam : public Optimizer<Adam> {
private:
    template <auto N>
    auto Init(auto &grads) -> void {
        [&grads, this]<auto... Is>(std::index_sequence<Is...>) {
            m_ = std::make_tuple((Eval(std::get<Is>(grads).constant(0)))...);
            v_ = std::make_tuple((Eval(std::get<Is>(grads).constant(0)))...);
        }(std::make_index_sequence<N>());
    }
    template <auto N>
    auto WeightDecay(auto &params, auto &grads) -> void {
        [&params, &grads, this]<auto... Is>(std::index_sequence<Is...>) {
            ((std::get<Is>(grads) += weight_decay_ * std::get<Is>(params)),
             ...);
        }(std::make_index_sequence<N>());
    }
    template <auto N>
    auto StepImpl(auto &params, auto &grads) -> void {
        [&params, &grads, this]<auto... Is>(std::index_sequence<Is...>) {
            auto &m = std::any_cast<decltype(std::make_tuple(
                (Eval(std::get<Is>(grads).constant(0)))...)) &>(m_);
            auto &v = std::any_cast<decltype(std::make_tuple(
                (Eval(std::get<Is>(grads).constant(0)))...)) &>(v_);
            ((std::get<Is>(m) =
                  beta1_ * std::get<Is>(m) + (1 - beta1_) * std::get<Is>(grads),
              std::get<Is>(v) = beta2_ * std::get<Is>(v) +
                                (1 - beta2_) * std::get<Is>(grads).square(),
              std::get<Is>(params) -=
              learning_rate_ *
              (1 / (1 - std::pow(beta1_, iter_)) * std::get<Is>(m)) /
              ((1 / (1 - std::pow(beta2_, iter_)) * std::get<Is>(v)).sqrt() +
               std::get<Is>(v).constant(eps_))),
             ...);
        }(std::make_index_sequence<N>());
    }
    float learning_rate_, beta1_, beta2_, eps_, weight_decay_;
    std::any m_, v_;
    int iter_ = 0;

public:
    Adam(float learning_rate = 1e-3,
         float beta1 = 0.9,
         float beta2 = 0.999,
         float eps = 1e-8,
         float weight_decay = 0)
        : learning_rate_(learning_rate),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps),
          weight_decay_(weight_decay) {}
    auto Step(auto &params, auto &grads) -> void {
        constexpr auto size = getSize(params);
        if (!m_.has_value()) {
            Init<size>(grads);
        }
        if (weight_decay_) {
            WeightDecay<size>(params, grads);
        }
        iter_++;
        StepImpl<size>(params, grads);
    }
};
}  // namespace optim
#endif  // OPTIMIZER_H