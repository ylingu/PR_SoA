#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
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

#endif  // UTILS_H