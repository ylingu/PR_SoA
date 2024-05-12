#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
/**
 * @class Normalization
 * @brief Base class for normalization strategies.
 */
class Normalization {
protected:
    bool parametersInitialized_ = false;

public:
    /**
     * @brief Normalize the features.
     * @param features The features to normalize.
     * @return The normalized features.
     */
    virtual auto Normalize(const Eigen::MatrixXd &features)
        -> Eigen::MatrixXd = 0;

    /**
     * @brief Default destructor.
     */
    virtual ~Normalization() = default;
};

/**
 * @class LinearNormalization
 * @brief Linear normalization strategy.
 */
class LinearNormalization : public Normalization {
private:
    Eigen::VectorXd max_;  // Maximum values
    Eigen::VectorXd min_;  // Minimum values
public:
    /**
     * @brief Normalize the features.
     * @param features The features to normalize.
     * @return The normalized features.
     */
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

/**
 * @class ZScoreStandardization
 * @brief Z-score standardization strategy.
 */
class ZScoreStandardization : public Normalization {
private:
    Eigen::VectorXd mean_;  // Mean values
    Eigen::VectorXd std_;   // Standard deviations
public:
    /**
     * @brief Normalize the features.
     * @param features The features to normalize.
     * @return The normalized features.
     */
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

/**
 * @brief Calculate the accuracy of the prediction.
 * @param predict The predicted labels.
 * @param label The true labels.
 * @return The accuracy.
 */
auto CalcAccuracy(const std::vector<int> &predict,
                  const std::vector<int> &label) -> double;

/**
 * @brief Creates an array of dimension pairs for Eigen tensor contraction
 * operation
 *
 * @param a Contraction dimension of the first tensor
 * @param b Contraction dimension of the second tensor
 * @return Eigen::array<Eigen::IndexPair<int>, 1> Array of dimension pairs for
 * contraction operation
 */
inline auto ProdDims(int a, int b) -> Eigen::array<Eigen::IndexPair<int>, 1> {
    return Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(a, b)};
}

/**
 * @brief Broadcasts a tensor to a higher dimension.
 *
 * @tparam Derived The type of the tensor to be broadcasted. Must be derived
 * from Eigen::TensorBase.
 * @tparam N The number of dimensions to broadcast to. Must be greater than or
 * equal to the number of dimensions of the input tensor.
 * @param x The tensor to be broadcasted.
 * @param dims The dimensions to broadcast to.
 * @param col Whether to broadcast the tensor to the column dimension.
 * @return Eigen::Tensor<typename Derived::Scalar, N> The broadcasted tensor.
 *
 * @note The function will throw an error if N is less than the number of
 * dimensions of the input tensor.
 */
template <typename Derived, int N>
    requires std::is_base_of_v<Eigen::TensorBase<Derived>, Derived> &&
                 (N >= Derived::NumDimensions)
auto Broadcast(const Derived &x,
               const Eigen::DSizes<long, N> &dims,
               bool col = false) -> Eigen::Tensor<typename Derived::Scalar, N> {
    Eigen::array<int, N> cast;
    cast.fill(1);
    if (col && x.NumDimensions == 1) {
        std::copy_n(x.dimensions().begin(), 1, cast.end() - 2);
        Eigen::Tensor<typename Derived::Scalar, N> x_reshaped = x.reshape(cast);
        std::copy_n(dims.begin(), N, cast.begin());
        cast[N - 2] = x.dimension(0);
        return x_reshaped.broadcast(cast);
    }
    std::copy_n(
        x.dimensions().begin(), x.NumDimensions, cast.end() - x.NumDimensions);
    Eigen::Tensor<typename Derived::Scalar, N> x_reshaped = x.reshape(cast);
    cast.fill(1);
    std::copy_n(dims.begin(), N - x.NumDimensions, cast.begin());
    return x_reshaped.broadcast(cast);
};

#endif  // UTILS_H