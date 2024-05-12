#ifndef LAYERS_H
#define LAYERS_H


#include <unsupported/Eigen/CXX11/Tensor>

#include "utils.h"

namespace nn {

template <typename Derived>
    requires std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>
class Relu {
private:
    Eigen::Tensor<bool, Derived::NumIndices> mask_;
    Derived zeros_;

public:
    auto Forward(const Derived &x) -> Derived {
        zeros_ = x.constant(0);
        mask_ = x <= zeros_;
        return x.cwiseMax(zeros_);
    }
    auto Backward(const Derived &dout) -> Derived {
        return mask_.select(zeros_, dout);
    }
};

template <typename Derived>
    requires std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>
class Sigmoid {
private:
    Derived out_;

public:
    auto Forward(const Derived &x) -> Derived {
        out_ = 1 / (1 + (-x).exp());
        return out_;
    }
    auto Backward(const Derived &dout) -> Derived {
        return out_ * (1 - out_) * dout;
    }
};

template <typename Derived>
    requires std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>
class Linear {
private:
    Eigen::Tensor<typename Derived::Scalar, 2> w_, dw_;
    Eigen::Tensor<typename Derived::Scalar, 1> b_, db_;
    Derived x_;

public:
    Linear(int in_features, int out_features)
        : w_(in_features, out_features),
          b_(out_features),
          dw_(in_features, out_features),
          db_(out_features) {
        w_.template setRandom();
        w_ = 2 / sqrt(in_features) * (w_ - static_cast<Derived::Scalar>(0.5));
        b_.template setRandom();
        b_ = 2 / sqrt(in_features) * (b_ - static_cast<Derived::Scalar>(0.5));
    }
    auto Forward(const Derived &x) -> Derived {
        x_ = x;
        Derived contracted = x.contract(w_, ProdDims(x.NumDimensions - 1, 0));
        return contracted + Broadcast(b_, x.dimensions());
    }
    auto Backward(const Derived &dout) -> Derived {
        decltype(w_) dout_flat = dout.reshape(Eigen::array<int, 2>{
                         dout.size() / dout.dimension(dout.NumDimensions - 1),
                         dout.dimension(dout.NumDimensions - 1)}),
                     x_flat = x_.reshape(Eigen::array<int, 2>{
                         x_.size() / x_.dimension(x_.NumDimensions - 1),
                         x_.dimension(x_.NumDimensions - 1)});
        dw_ = x_flat.contract(dout_flat, ProdDims(0, 0));
        db_ = dout_flat.sum(Eigen::array<int, 1>{0});
        return dout.contract(w_, ProdDims(dout.NumIndices - 1, 1));
    }
    auto dw() -> decltype((dw_)) { return dw_; }
    auto db() -> decltype((db_)) { return db_; }
};

template <typename DerivedY, typename DerivedT>
    requires std::is_base_of_v<Eigen::TensorBase<DerivedY>, DerivedY> &&
             std::is_base_of_v<Eigen::TensorBase<DerivedT>, DerivedT>
class CrossEntropyLoss {
private:
    DerivedY y_, t_;
    int batch_size = 1;

public:
    auto Forward(const DerivedY &y, const DerivedT &t) ->
        typename DerivedY::Scalar {
        // 计算softmax
        auto max = Broadcast(y.maximum(Eigen::array<int, 1>{y.NumIndices - 1}),
                             y.dimensions(),
                             true);
        auto exp = (y - max).exp();
        auto sum = Broadcast(exp.sum(Eigen::array<int, 1>{y.NumIndices - 1}),
                             y.dimensions(),
                             true);
        y_ = exp / sum;
        // 计算交叉熵
        if (y.NumDimensions != t.NumDimensions) {
            t_ = y.constant(0);
            if (y.NumIndices == 2) {
                batch_size = y.dimension(0);
                for (int i = 0; i != t.size(); ++i) {
                    t_(i, t(i)) = 1;
                }
            } else
                t_(static_cast<int>(t())) = 1;
        }
        return -(t_ * (y_ + y_.constant(1e-7)).log()).sum() / batch_size;
    }
    auto Backward() -> DerivedY { return (y_ - t_) / batch_size; }
};
}  // namespace nn


#endif  // LAYERS_H