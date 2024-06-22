#ifndef LAYERS_H
#define LAYERS_H

#include <any>
#include <print>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include <variant>

#include "optimizer.h"
#include "utils.h"

namespace nn {

template <typename T>
concept Optimizer = std::is_base_of_v<optim::Optimizer<T>, T>;

template <typename T>
concept HasParams = requires(T t) { t.Params(); };

template <typename T>
concept HasGrads = requires(T t) { t.Grads(); };

template <typename T>
concept Formattable = requires(T t) { t.ToString(); };

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
    auto ToString() const -> std::string {
        return static_cast<Derived *>(this)->ToString();
    }
};

template <EigenTensor Tensor>
class ReLU : public Module<ReLU<Tensor>> {
private:
    Eigen::Tensor<bool, Tensor::NumIndices> mask_;
    Tensor zeros_;

public:
    auto Forward(const Tensor &x) -> Tensor {
        zeros_ = x.constant(0);
        mask_ = x <= zeros_;
        return x.cwiseMax(zeros_);
    }
    auto Backward(const Tensor &dout) -> Tensor {
        return mask_.select(zeros_, dout);
    }
    auto ToString() const -> std::string { return "ReLU()"; }
};

template <EigenTensor Tensor>
class Sigmoid : public Module<Sigmoid<Tensor>> {
private:
    Tensor out_;

public:
    auto Forward(const Tensor &x) -> Tensor {
        out_ = 1 / (1 + (-x).exp());
        return out_;
    }
    auto Backward(const Tensor &dout) -> Tensor {
        return out_ * (1 - out_) * dout;
    }
    auto ToString() const -> std::string { return "Sigmoid()"; }
};

template <EigenTensor Tensor>
class Flatten : public Module<Flatten<Tensor>> {
private:
    Eigen::array<long, Tensor::NumIndices> dims_;

public:
    auto Forward(const Tensor &x) -> Eigen::Tensor<typename Tensor::Scalar, 2> {
        dims_ = x.dimensions();
        return x.reshape(Eigen::array<long, 2>{dims_[0], x.size() / dims_[0]});
    }
    auto Backward(const Eigen::Tensor<typename Tensor::Scalar, 2> &dout)
        -> Tensor {
        return dout.reshape(dims_);
    }
    auto ToString() const -> std::string {
        return "Flatten(start_dim=1, end_dim=-1)";
    }
};

template <EigenTensor Tensor>
class Linear : public Module<Linear<Tensor>> {
private:
    Eigen::Tensor<typename Tensor::Scalar, 2> w_, dw_;
    Eigen::Tensor<typename Tensor::Scalar, 1> b_, db_;
    Tensor x_;

public:
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
        decltype(w_) dout_flat = dout.reshape(Eigen::array<long, 2>{
                         dout.size() / dout.dimension(dout.NumIndices - 1),
                         dout.dimension(dout.NumIndices - 1)}),
                     x_flat = x_.reshape(Eigen::array<long, 2>{
                         x_.size() / x_.dimension(x_.NumIndices - 1),
                         x_.dimension(x_.NumIndices - 1)});
        dw_ = x_flat.contract(dout_flat, ProdDims(0, 0));
        db_ = dout_flat.sum(Eigen::array<int, 1>{0});
        return dout.contract(w_, ProdDims(dout.NumIndices - 1, 1));
    }
    auto Params() -> std::tuple<decltype(w_) &, decltype(b_) &> {
        return {w_, b_};
    }
    auto Grads() -> std::tuple<decltype(dw_) &, decltype(db_) &> {
        return {dw_, db_};
    }
    auto ToString() const -> std::string {
        return std::format("Linear(in_features={}, out_features={}, bias=True)",
                           w_.dimension(0),
                           w_.dimension(1));
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

template <typename... Modules>
    requires(std::conjunction_v<std::is_base_of<Module<Modules>, Modules>...>)
class Sequential {
private:
    template <auto N>
    auto ForwardImpl(const auto &x) const {
        if constexpr (N) {
            return std::get<N - 1>(modules_)->Forward(ForwardImpl<N - 1>(x));
        } else {
            return x;
        }
    }
    template <auto N>
    auto BackwardImpl(const auto &dout) const {
        if constexpr (N) {
            BackwardImpl<N - 1>(std::get<N - 1>(modules_)->Backward(dout));
        }
    }
    template <auto N>
    auto ToStringImpl() const {
        if constexpr (N) {
            return std::format("{}  ({}): {}\n",
                               ToStringImpl<N - 1>(),
                               N - 1,
                               std::get<N - 1>(modules_)->ToString());
        }
        return std::string();
    }
    template <size_t I>
    auto GetParams(const auto &params) const {
        if constexpr (I < sizeof...(Modules)) {
            auto module = std::ref(std::get<I>(modules_));
            if constexpr (HasParams<decltype(*(module.get()))>) {
                return GetParams<I + 1>(
                    std::tuple_cat(params, module.get()->Params()));
            } else {
                return GetParams<I + 1>(params);
            }
        } else {
            return params;
        }
    }
    template <size_t I>
    auto GetGrads(const auto &grads) const {
        if constexpr (I < sizeof...(Modules)) {
            auto module = std::ref(std::get<I>(modules_));
            if constexpr (HasGrads<decltype(*(module.get()))>) {
                return GetGrads<I + 1>(
                    std::tuple_cat(grads, module.get()->Grads()));
            } else {
                return GetGrads<I + 1>(grads);
            }
        } else {
            return grads;
        }
    }
    std::tuple<std::unique_ptr<Modules>...> modules_;
    std::any params_, grads_;

public:
    Sequential(Modules &&...modules)
        : modules_{std::make_unique<Modules>(
              std::forward<Modules>(modules))...},
          params_(this->GetParams<0>(std::tuple<>())),
          grads_(this->GetGrads<0>(std::tuple<>())) {}

    auto Forward(const auto &x) const {
        return ForwardImpl<sizeof...(Modules)>(x);
    }
    auto Backward(const auto &dout) const -> void {
        BackwardImpl<sizeof...(Modules)>(dout);
    }
    template <Optimizer Optimizer>
    auto Step(Optimizer &optimizer) const -> void {
        optimizer.Step(
            *std::any_cast<decltype(this->GetParams<0>(std::tuple<>{}))>(
                &params_),
            *std::any_cast<decltype(this->GetGrads<0>(std::tuple<>{}))>(
                &grads_));
    }
    auto ToString() const -> std::string {
        return "Sequential(\n" + ToStringImpl<sizeof...(Modules)>() + ")\n";
    }
};
}  // namespace nn

template <nn::Formattable T>
struct std::formatter<T> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const T &f, FormatContext &ctx) const {
        return format_to(ctx.out(), "{}", f.ToString());
    }
};
#endif  // LAYERS_H