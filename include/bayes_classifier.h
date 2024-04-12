#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

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
    Eigen::MatrixXd mean_;     ///< The mean of the training data.
    Eigen::MatrixXd cov_;      ///< The covariance of the training data.
    Eigen::VectorXd prior_;    ///< The prior probabilities of the classes.
    Preprocess preprocessor_;  ///< The preprocessor for the data.

public:
    /**
     * @brief Constructs a new BayesClassifier object.
     *
     * @param strategy The normalization strategy to use.
     */
    BayesClassifier(std::unique_ptr<Normalization> strategy)
        : preprocessor_(std::move(strategy)) {}

    /**
     * @brief Trains the Bayes classifier with the given training data and
     * labels.
     *
     * @param train_data The training data, a vector of cv::Mat objects.
     * @param train_label The labels for the training data, a vector of
     * integers.
     * @param class_num The number of classes in the data.
     */
    auto Train(const std::vector<cv::Mat> &train_data,
               const std::vector<int> &train_label, const int &class_num)
        -> void;

    /**
     * @brief Predicts the labels for the given test data.
     *
     * @param test_data The test data, a vector of cv::Mat objects.
     * @return A vector of predicted labels for the test data.
     */
    auto Predict(const std::vector<cv::Mat> &test_data) const
        -> std::vector<int>;
};

#endif  // BAYESCLASSIFIER_H