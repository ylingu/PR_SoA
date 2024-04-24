#include "bayes_classifier.h"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

auto BayesClassifier::Preprocess(const std::vector<cv::Mat> &data,
                                 const int pca_dims) -> Eigen::MatrixXd {
    auto features = feature_extractor_->BatchExtract(data);
    if (normalize_strategy_ != nullptr) {
        features = normalize_strategy_->Normalize(features);
    }
    if (pca_dims) {
        assert(pca_dims > 0 && pca_dims <= features.cols());
        cv::Mat pca_data;
        cv::eigen2cv(features, pca_data);
        if (pca_eigen_vector_.empty()) {
            cv::PCA pca(pca_data, cv::Mat(), cv::PCA::DATA_AS_ROW, pca_dims);
            cv::transpose(pca.eigenvectors, pca_eigen_vector_);
        }
        cv::Mat pca_result = pca_data * pca_eigen_vector_;
        cv::cv2eigen(pca_result, features);
    }
    return features;
}

auto BayesClassifier::Train(const Eigen::MatrixXd &train_data,
                            const std::vector<int> &train_label) -> void {
    int n = train_data.rows();
    // * Calculate the mean, covariance, and prior probability of each class
    int feature_num = train_data.row(0).size();
    mean_ = Eigen::MatrixXd::Zero(class_num_, feature_num);
    cov_ = Eigen::MatrixXd::Zero(class_num_, feature_num);
    prior_ = Eigen::VectorXd::Zero(class_num_);
    for (int i = 0; i < n; ++i) {
        mean_.row(train_label[i]) += train_data.row(i);
        prior_(train_label[i])++;
    }
    for (int i = 0; i < class_num_; ++i) {
        mean_.row(i) /= prior_(i);
    }
    for (int i = 0; i < n; ++i) {
        cov_.row(train_label[i]) +=
            (train_data.row(i) - mean_.row(train_label[i]))
                .cwiseProduct(train_data.row(i) - mean_.row(train_label[i]));
    }
    for (int i = 0; i < class_num_; ++i) {
        cov_.row(i) /= prior_(i);
    }
    prior_ /= n;
}

auto BayesClassifier::Predict(const Eigen::MatrixXd &test_data) const
    -> std::vector<int> {
    int n = test_data.rows();
    auto res = std::vector<int>();
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(mean_.rows());
        for (int j = 0; j < mean_.rows(); ++j) {
            prob(j) = -0.5 * (test_data.row(i) - mean_.row(j))
                                 .cwiseProduct(cov_.row(j).cwiseInverse())
                                 .cwiseProduct(test_data.row(i) - mean_.row(j))
                                 .sum() -
                      0.5 * log(cov_.row(j).prod()) + log(prior_(j));
        }
        int max_index;
        prob.maxCoeff(&max_index);
        res.push_back(max_index);
    }
    return res;
}
