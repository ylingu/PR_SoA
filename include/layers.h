#ifndef LAYERS_H
#define LAYERS_H

#include <any>
#include <unsupported/Eigen/CXX11/Tensor>
#include <variant>

#include "optimizer.h"
#include "utils.h"

namespace nn {

template <typename Derived>
class Module {
public:
    virtual ~Module() = default;
    template <EigenTensor Tensor>
    auto Forward(const Tensor &x) -> Tensor {
        return static_cast<Derived *>(this)->Forward(x);
    }
    template <EigenTensor Tensor>
    auto Backward(const Tensor &dout) -> Tensor {
        return static_cast<Derived *>(this)->Backward(dout);
    }
};

template <EigenTensor Tensor>
class Relu : public Module<Relu<Tensor>> {
private:
    Eigen::Tensor<bool, Tensor::NumIndices> mask_;
    Tensor zeros_;

public:
    using type = Tensor;
    auto Forward(const Tensor &x) -> Tensor {
        zeros_ = x.constant(0);
        mask_ = x <= zeros_;
        return x.cwiseMax(zeros_);
    }
    auto Backward(const Tensor &dout) -> Tensor {
        return mask_.select(zeros_, dout);
    }
};

template <EigenTensor Tensor>
class Sigmoid : public Module<Sigmoid<Tensor>> {
private:
    Tensor out_;

public:
    using type = Tensor;
    auto Forward(const Tensor &x) -> Tensor {
        out_ = 1 / (1 + (-x).exp());
        return out_;
    }
    auto Backward(const Tensor &dout) -> Tensor {
        return out_ * (1 - out_) * dout;
    }
};

template <EigenTensor Tensor>
class Flatten : public Module<Flatten<Tensor>> {
private:
    Eigen::array<int, Tensor::NumIndices> dims_;

public:
    using type = Tensor;
    auto Forward(const Tensor &x) -> Eigen::Tensor<typename Tensor::Scalar, 2> {
        dims_ = x.dimensions();
        return x.reshape(Eigen::array<int, 2>{dims_[0], x.size() / dims_[0]});
    }
    auto Backward(const Eigen::Tensor<typename Tensor::Scalar, 2> &dout)
        -> Tensor {
        return dout.reshape(dims_);
    }
};

template <typename T>
concept Optimizer = std::is_base_of_v<optim::Optimizer<T>, T>;

template <EigenTensor Tensor>
class Linear : public Module<Linear<Tensor>> {
private:
    Eigen::Tensor<typename Tensor::Scalar, 2> w_, dw_;
    Eigen::Tensor<typename Tensor::Scalar, 1> b_, db_;
    Tensor x_;

public:
    using type = Tensor;
    Linear(int in_features, int out_features)
        : w_(in_features, out_features),
          b_(out_features),
          dw_(in_features, out_features),
          db_(out_features) {
        w_.template setRandom();
        w_ = 2 / sqrt(in_features) * (w_ - static_cast<Tensor::Scalar>(0.5));
        b_.template setRandom();
        b_ = 2 / sqrt(in_features) * (b_ - static_cast<Tensor::Scalar>(0.5));
    }
    auto Forward(const Tensor &x) -> Tensor {
        x_ = x;
        Tensor contracted = x.contract(w_, ProdDims(x.NumIndices - 1, 0));
        return contracted + Broadcast(b_, x.dimensions());
    }
    auto Backward(const Tensor &dout) -> Tensor {
        decltype(w_) dout_flat = dout.reshape(Eigen::array<int, 2>{
                         dout.size() / dout.dimension(dout.NumIndices - 1),
                         dout.dimension(dout.NumIndices - 1)}),
                     x_flat = x_.reshape(Eigen::array<int, 2>{
                         x_.size() / x_.dimension(x_.NumIndices - 1),
                         x_.dimension(x_.NumIndices - 1)});
        dw_ = x_flat.contract(dout_flat, ProdDims(0, 0));
        db_ = dout_flat.sum(Eigen::array<int, 1>{0});
        return dout.contract(w_, ProdDims(dout.NumIndices - 1, 1));
    }
    template <Optimizer OptType>
    auto Update(const OptType &optimizer) -> void {
        optimizer.Step(w_, dw_);
        optimizer.Step(b_, db_);
    }
};

template <EigenTensor Tensor>
class CrossEntropyLoss {
private:
    Tensor y_, t_;
    int batch_size = 1;

public:
    auto Forward(
        const Tensor &y,
        const std::variant<Eigen::Tensor<int, 0>, Eigen::Tensor<int, 1>> &t) ->
        typename Tensor::Scalar {
        // 计算softmax
        Eigen::Tensor<typename Tensor::Scalar, y.NumIndices - 1> max =
            y.maximum(Eigen::array<int, 1>{y.NumIndices - 1});
        auto max_broadcasted = Broadcast(max, y.dimensions(), true);
        decltype(y) exp = (y - max_broadcasted).exp();
        Eigen::Tensor<typename Tensor::Scalar, y.NumIndices - 1> sum =
            exp.sum(Eigen::array<int, 1>{y.NumIndices - 1});
        auto sum_broadcasted = Broadcast(sum, y.dimensions(), true);
        y_ = exp / sum_broadcasted;
        // 计算交叉熵
        if (std::get_if<Eigen::Tensor<int, y.NumIndices - 1>>(&t) != nullptr) {
            t_ = y.constant(0);
            if (y.NumIndices == 2) {
                batch_size = y.dimension(0);
                for (int i = 0; i != std::get<Eigen::Tensor<int, 1>>(t).size();
                     ++i) {
                    t_(i, std::get<Eigen::Tensor<int, 1>>(t)(i)) = 1;
                }
            } else
                t_(std::get<Eigen::Tensor<int, 0>>(t)()) = 1;
        }
        Eigen::Tensor<typename Tensor::Scalar, 0> loss =
            -(t_ * (y_ + y_.constant(1e-7)).log()).sum();
        return loss() / batch_size;
    }
    auto Backward() -> Tensor { return (y_ - t_) / y_.constant(batch_size); }
};

template <typename T>
concept HasUpdate = requires(T t) { t.Update(std::declval<optim::SGD>()); };

template <typename... Modules>
    requires(std::is_base_of_v<Module<Modules>, Modules> && ...)
class Sequential {
private:
    template <size_t I>
    using Tensor =
        typename std::tuple_element_t<I, std::tuple<Modules...>>::type;
    template <std::size_t... Is>
    auto ForwardImpl(const Tensor<0> &x, std::index_sequence<Is...>)
        -> Tensor<sizeof...(Modules) - 1> {
        std::any y = x;
        ((y = std::get<Is>(modules_)->Forward(std::any_cast<Tensor<Is>>(y))),
         ...);
        return std::any_cast<Tensor<sizeof...(Modules) - 1>>(y);
    }
    template <std::size_t... Is>
    auto BackwardImpl(const Tensor<sizeof...(Modules) - 1> &dout,
                      std::index_sequence<Is...>) -> void {
        std::any d = dout;
        ((d =
              [module =
                   std::ref(std::get<sizeof...(Modules) - 1 - Is>(modules_)),
               &d]() {
                  using OutputType = decltype(module.get()->Forward(
                      std::declval<Tensor<sizeof...(Modules) - 1 - Is>>()));
                  return module.get()->Backward(std::any_cast<OutputType>(d));
              }()),
         ...);
    }
    template <Optimizer OptType, std::size_t... Is>
    auto StepImpl(const OptType &optimizer,
                  std::index_sequence<Is...>) -> void {
        (([module = std::ref(std::get<sizeof...(Modules) - 1 - Is>(modules_)),
           &optimizer]() {
             if constexpr (HasUpdate<decltype(*(module.get()))>) {
                 module.get()->Update(optimizer);
             }
         }()),
         ...);
    }

    std::tuple<std::unique_ptr<Modules>...> modules_;

public:
    Sequential(Modules &&...modules)
        : modules_{
              std::make_unique<Modules>(std::forward<Modules>(modules))...} {}

    auto Forward(const Tensor<0> &x) -> Tensor<sizeof...(Modules) - 1> {
        return ForwardImpl(x, std::make_index_sequence<sizeof...(Modules)>{});
    }
    auto Backward(const Tensor<sizeof...(Modules) - 1> &dout) -> void {
        BackwardImpl(dout, std::make_index_sequence<sizeof...(Modules)>{});
    }
    template <Optimizer Optimizer>
    auto Step(const Optimizer &optimizer) -> void {
        StepImpl(optimizer, std::make_index_sequence<sizeof...(Modules)>{});
    }
};
}  // namespace nn

#endif  // LAYERS_H