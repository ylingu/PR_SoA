#include "bayes_classifier.h"

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"

auto BayesClassifier::Train(const std::vector<cv::Mat> &train_data,
                            const std::vector<int> &train_label,
                            const int &class_num) -> void {
    int n = train_data.size();
    auto features = preprocessor_.Preprocessing(train_data);
    // * Calculate the mean, covariance, and prior probability of each class
    int feature_num = features.row(0).size();
    mean_ = Eigen::MatrixXd::Zero(class_num, feature_num);
    cov_ = Eigen::MatrixXd::Zero(class_num, feature_num);
    prior_ = Eigen::VectorXd::Zero(class_num);
    for (int i = 0; i < n; ++i) {
        mean_.row(train_label[i]) += features.row(i);
        prior_(train_label[i])++;
    }
    for (int i = 0; i < class_num; ++i) {
        mean_.row(i) /= prior_(i);
    }
    for (int i = 0; i < n; ++i) {
        cov_.row(train_label[i]) +=
            (features.row(i) - mean_.row(train_label[i]))
                .cwiseProduct(features.row(i) - mean_.row(train_label[i]));
    }
    for (int i = 0; i < class_num; ++i) {
        cov_.row(i) /= prior_(i);
    }
    prior_ /= n;
}

auto BayesClassifier::Predict(const std::vector<cv::Mat> &test_data) const
    -> std::vector<int> {
    int n = test_data.size();
    auto features = preprocessor_.Preprocessing(test_data);
    auto res = std::vector<int>();
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(mean_.rows());
        for (int j = 0; j < mean_.rows(); ++j) {
            prob(j) = -0.5 * (features.row(i) - mean_.row(j))
                                 .cwiseProduct(cov_.row(j).cwiseInverse())
                                 .cwiseProduct(features.row(i) - mean_.row(j))
                                 .sum() -
                      0.5 * log(cov_.row(j).prod()) + log(prior_(j));
        }
        int max_index;
        prob.maxCoeff(&max_index);
        res.push_back(max_index);
    }
    return res;
}
