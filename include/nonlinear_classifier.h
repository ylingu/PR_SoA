#ifndef NONLINEAR_CLASSIFIER_H
#define NONLINEAR_CLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "csv.h"
#include "utils.h"

/**
 * @struct TreeNode
 * @brief A node in the decision tree.
 * @tparam T The type of data stored in the node.
 */
template <typename T>
struct TreeNode {
    T data;                              ///< The data stored in the node.
    int split;                           ///< The split criterion.
    int label;                           ///< The label of the node.
    std::unique_ptr<TreeNode<T>> left;   ///< The left child node.
    std::unique_ptr<TreeNode<T>> right;  ///< The right child node.

    /**
     * @brief Construct a new TreeNode object
     * @param data The data to be stored in the node.
     * @param split The split criterion.
     * @param label The label of the node.
     */
    TreeNode(T data, int split, int label)
        : data(data),
          split(split),
          label(label),
          left(nullptr),
          right(nullptr) {}
};

/**
 * @class KNNClassifier
 * @brief A k-nearest neighbors classifier.
 */
class KNNClassifier {
public:
    /**
     * @brief Construct a new KNNClassifier object
     * @param k The number of neighbors to consider.
     */
    KNNClassifier(int k) : k_(k) {}

    /**
     * @brief Fit the model to the training data.
     * @param train_data The training data.
     */
    auto Fit(const std::vector<std::pair<Eigen::VectorXd, int>> &train_data)
        -> void;

    /**
     * @brief Predict the labels of the test data.
     * @param test_data The test data.
     * @return The predicted labels.
     */
    auto Predict(const std::vector<Eigen::VectorXd> &test_data) const
        -> std::vector<int>;

private:
    int k_;  ///< The number of neighbors to consider.
    std::unique_ptr<TreeNode<Eigen::VectorXd>>
        root_;  ///< The root of the decision tree.
};

/**
 * @class SVMClassifier
 * @brief A support vector machine classifier.
 */
class SVMClassifier {
private:
    cv::Ptr<cv::ml::SVM> svm_;  ///< The SVM model.
    static int kEpochs;         ///< The number of training epochs.
    static double kEpsilon;     ///< The tolerance for stopping criterion.

public:
    Preprocess preprocessor_;  ///< The preprocessing strategy.

    /**
     * @brief Construct a new SVMClassifier object
     * @param strategy The normalization strategy.
     */
    SVMClassifier(std::unique_ptr<Normalization> strategy)
        : preprocessor_(std::move(strategy)) {}

    /**
     * @brief Train the model.
     * @param train_data The training data.
     * @param train_label The labels of the training data.
     * @param validate_data The validation data.
     * @param validate_label The labels of the validation data.
     * @param c_range The range of the C parameter.
     * @param gamma_range The range of the gamma parameter.
     * @param class_num The number of classes.
     */
    auto Train(const std::vector<cv::Mat> &train_data,
               const std::vector<int> &train_label,
               const std::vector<cv::Mat> &validate_data,
               const std::vector<int> &validate_label,
               const std::vector<double> &c_range,
               const std::vector<double> &gamma_range, const int &class_num)
        -> void;

    /**
     * @brief Predict the labels of the test data.
     * @param test_data_mat The test data.
     * @param svm The SVM model. If not provided, use the trained model.
     * @return The predicted labels.
     */
    auto Predict(const cv::Mat &test_data_mat,
                 const std::optional<cv::Ptr<cv::ml::SVM>> &svm = std::nullopt)
        -> std::vector<int>;

    /**
     * @brief Save the trained model.
     * @param filepath The path to save the model. Default is "model.xml".
     */
    auto SaveModel(const std::string &filepath = "model.xml") -> void;

    /**
     * @brief Load a trained model.
     * @param filepath The path to load the model from. Default is "model.xml".
     */
    auto LoadModel(const std::string &filepath = "model.xml") -> void;
};

#endif  // NONLINEAR_CLASSIFIER_H