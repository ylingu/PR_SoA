#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class GLCM
 * @brief Grey Level Co-occurrence Matrix (GLCM) feature extraction.
 */
class GLCM {
private:
    static int kLevel;  // Number of grey levels
    static int kD;      // Distance
public:
    /**
     * @brief Compress the image.
     * @param img The image to compress.
     * @return The compressed image.
     */
    static auto Compress(const cv::Mat &img) -> cv::Mat;

    /**
     * @brief Get the GLCM of the image.
     * @param img The image.
     * @param angle The angle.
     * @return The GLCM.
     */
    static auto GetGLCM(const cv::Mat &img, const double &angle)
        -> Eigen::MatrixXd;

    /**
     * @brief Get the GLCM features.
     * @param glcm The GLCM.
     * @return The GLCM features.
     */
    static auto GetFeature(const Eigen::MatrixXd &glcm) -> Eigen::VectorXd;
};

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
 * @class Preprocess
 * @brief Preprocessing class.
 */
class Preprocess {
private:
    static std::vector<double> kAngles;        // Angles
    std::unique_ptr<Normalization> strategy_;  // Normalization strategy
public:
    /**
     * @brief Construct a new Preprocess object.
     * @param strategy The normalization strategy.
     */
    Preprocess(std::unique_ptr<Normalization> strategy)
        : strategy_(std::move(strategy)) {}

    /**
     * @brief Preprocess the data.
     * @param data The data to preprocess.
     * @return The preprocessed data.
     */
    auto Preprocessing(const std::vector<cv::Mat> &data) const
        -> Eigen::MatrixXd;
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