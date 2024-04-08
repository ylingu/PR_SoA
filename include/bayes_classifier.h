#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"

// 贝叶斯分类器
class BayesClassifier {
private:
    Eigen::MatrixXd mean_;   // 均值
    Eigen::MatrixXd cov_;    // 协方差
    Eigen::VectorXd prior_;  // 先验概率
    Preprocess preprocessor_;

public:
    BayesClassifier(std::unique_ptr<Normalization> strategy)
        : preprocessor_(std::move(strategy)) {}
    auto Train(const std::vector<cv::Mat> &train_data,
               const std::vector<int> &train_label, const int &class_num)
        -> void;
    auto Predict(const std::vector<cv::Mat> &test_data) const
        -> std::vector<int>;
};
#endif