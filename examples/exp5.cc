#include <print>

#include "layers.h"
using namespace std;

template <typename T>
void printTensor(const Eigen::Tensor<T, 4> &tensor) {
    for (int i = 0; i < tensor.dimension(0); ++i) {
        std::cout << "[";
        for (int j = 0; j < tensor.dimension(1); ++j) {
            std::cout << "[";
            for (int k = 0; k < tensor.dimension(2); ++k) {
                std::cout << "[";
                for (int l = 0; l < tensor.dimension(3); ++l) {
                    std::cout << tensor(i, j, k, l);
                    if (l != tensor.dimension(3) - 1) std::cout << ", ";
                }
                std::cout << "]";
                if (k != tensor.dimension(2) - 1) std::cout << ",\n";
            }
            std::cout << "]";
            if (j != tensor.dimension(1) - 1) std::cout << ",\n";
        }
        std::cout << "]";
        if (i != tensor.dimension(0) - 1) std::cout << ",\n";
    }
    std::cout << "\n\n";
}

int main() {
    auto layer = nn::Linear<Eigen::Tensor<float, 4>>(4, 5);
    Eigen::Tensor<float, 4> x(1, 2, 3, 4);
    x.setRandom();
    Eigen::Tensor<float, 4> y = layer.Forward(x);
    printTensor(y);
    Eigen::Tensor<float, 4> dout(1, 2, 3, 5);
    dout.setRandom();
    Eigen::Tensor<float, 4> dx = layer.Backward(dout);
    printTensor(dx);
    cout << layer.dw() << endl;
    cout << layer.db() << endl;
    return 0;
}