#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Tensor>
concept EigenTensor = std::is_base_of_v<Eigen::TensorBase<Tensor>, Tensor>;

namespace optim {
    
template <typename Derived>
class Optimizer {
public:
    virtual ~Optimizer() = default;
    template <EigenTensor Tensor>
    auto Step(Tensor &param, const Tensor &grad) -> void {
        static_cast<Derived *>(this)->Step(param, grad);
    }
};
class SGD : public Optimizer<SGD> {
private:
    float learning_rate_;

public:
    SGD(float learning_rate) : learning_rate_(learning_rate) {}
    template <EigenTensor Tensor>
    auto Step(Tensor &param, const Tensor &grad) const -> void {
        param -= learning_rate_ * grad;
    }
};
}  // namespace optim
#endif  // OPTIMIZER_H