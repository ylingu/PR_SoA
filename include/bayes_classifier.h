#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "feature_extraction.h"
#include "utils.h"

/**
 * @class BayesClassifier
 * @brief A class for a Bayes classifier.
 *
 * This class provides functions for training and predicting with a Bayes
 * classifier.
 */
class BayesClassifier {
private:
    int class_num_;          ///< Number of classes in the training data.
    Eigen::MatrixXd mean_;   ///< Mean of the training data.
    Eigen::MatrixXd cov_;    ///< Covariance of the training data.
    Eigen::VectorXd prior_;  ///< Prior probabilities of the classes.
    std::unique_ptr<Normalization>
        normalize_strategy_;  ///< Strategy for data normalization.
    std::unique_ptr<FeatureExtraction>
        feature_extractor_;     ///< Strategy for feature extraction.
    cv::Mat pca_eigen_vector_;  ///< Eigen vector of PCA

public:
    /**
     * @brief Constructs a BayesClassifier with optional normalization and
     * feature extraction strategies.
     *
     * This constructor initializes a BayesClassifier. If the normalization
     * strategy is not provided, it defaults to nullptr, indicating no
     * normalization will be applied. If the feature extraction strategy is not
     * provided, it defaults to an instance of GLCMFeatureExtraction.
     *
     * @param class_num The number of classes in the data.
     * @param normalize_strategy Unique pointer to a Normalization strategy
     * object. If nullptr, no normalization is applied (default is nullptr).
     * @param feature_extractor Unique pointer to a FeatureExtraction strategy
     * object. Defaults to GLCMFeatureExtraction if not provided.
     */
    BayesClassifier(
        const int class_num,
        std::unique_ptr<Normalization> normalize_strategy_ = nullptr,
        std::unique_ptr<FeatureExtraction> feature_extractor =
            std::make_unique<GLCMFeatureExtraction>())
        : class_num_(class_num),
          normalize_strategy_(std::move(normalize_strategy_)),
          feature_extractor_(std::move(feature_extractor)) {}

    /**
     * @brief Preprocesses the input data by extracting features and applying
     * normalization.
     *
     * This method processes a batch of images by first extracting features
     * using the configured feature extraction strategy, and then optionally
     * applying normalization if a normalization strategy has been set. The
     * method returns a matrix where each row corresponds to the feature vector
     * of an image from the input batch.
     *
     * @param data A vector of cv::Mat objects representing the input images.
     * @param pca_dims The target number of dimensions after PCA reduction. A
     * value of 0 indicates that PCA reduction should not be applied.
     * @return An Eigen::MatrixXd where each row is the feature vector of the
     * corresponding input image. The number of columns in the matrix
     * corresponds to the number of features extracted.
     */
    auto Preprocess(const std::vector<cv::Mat> &data,
                    const int pca_dims = 0) -> Eigen::MatrixXd;

    /**
     * @brief Trains the Bayes classifier with the given training data and
     * labels.
     *
     * @param train_data An Eigen::MatrixXd where each row represents a training
     * example and each column represents a feature.
     * @param train_label A std::vector<int> containing the class labels for
     * each training example. The size of this vector should match the number of
     * rows in train_data.
     */
    auto Train(const Eigen::MatrixXd &train_data,
               const std::vector<int> &train_label) -> void;

    /**
     * @brief Predicts the labels for the given test data.
     *
     * @param test_data An Eigen::MatrixXd where each row represents a test
     * example and each column represents a feature.
     * @return A std::vector<int> containing the predicted class labels for each
     * test example. The size of this vector will match the number of rows in
     * test_data.
     */
    auto Predict(const Eigen::MatrixXd &test_data) const -> std::vector<int>;
};

#endif  // BAYESCLASSIFIER_H