#ifndef NONLINEAR_CLASSIFIER_H
#define NONLINEAR_CLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"

template <typename T>
struct TreeNode {
    T data;
    int split;
    int label;
    std::unique_ptr<TreeNode<T>> left;
    std::unique_ptr<TreeNode<T>> right;
    TreeNode(T data, int split, int label)
        : data(data),
          split(split),
          label(label),
          left(nullptr),
          right(nullptr) {}
};

class KNNClassifier {
public:
    KNNClassifier(int k) : k_(k) {}
    auto Fit(const std::vector<std::pair<Eigen::VectorXd, int>> &train_data)
        -> void;
    auto Predict(const std::vector<Eigen::VectorXd> &test_data) const
        -> std::vector<int>;

private:
    int k_;
    std::unique_ptr<TreeNode<Eigen::VectorXd>> root_;
};

class SVMClassifier {
private:
    cv::Ptr<cv::ml::SVM> svm_;
    static int kEpochs;
    static double kEpsilon;

public:
    Preprocess preprocessor_;

    SVMClassifier(std::unique_ptr<Normalization> strategy)
        : preprocessor_(std::move(strategy)) {}
    auto Train(const std::vector<cv::Mat> &train_data,
               const std::vector<int> &train_label,
               const std::vector<cv::Mat> &validate_data,
               const std::vector<int> &validate_label,
               const std::vector<double> &c_range,
               const std::vector<double> &gamma_range, const int &class_num)
        -> void;
    auto Predict(const cv::Mat &test_data_mat,
                 const std::optional<cv::Ptr<cv::ml::SVM>> &svm = std::nullopt)
        -> std::vector<int>;
    auto SaveModel(const std::string &filepath = "model.xml") -> void;
    auto LoadModel(const std::string &filepath = "model.xml") -> void;
};

#endif