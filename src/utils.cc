#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

auto LinearNormalization::Normalize(const Eigen::MatrixXd &features)
    -> Eigen::MatrixXd {
    if (!parametersInitialized_) {
        max_ = features.colwise().maxCoeff();
        min_ = features.colwise().minCoeff();
        parametersInitialized_ = true;
    }
    return (features.rowwise() - min_.transpose()).array().rowwise() /
           (max_ - min_).transpose().array();
}

auto ZScoreStandardization::Normalize(const Eigen::MatrixXd &features)
    -> Eigen::MatrixXd {
    if (!parametersInitialized_) {
        mean_ = features.colwise().mean();
        std_ = ((features.rowwise() - mean_.transpose())
                    .cwiseProduct(features.rowwise() - mean_.transpose()))
                   .colwise()
                   .sum()
                   .cwiseSqrt();
        parametersInitialized_ = true;
    }
    return (features.rowwise() - mean_.transpose()).array().rowwise() /
               (std_ * 6).transpose().array() +
           0.5;
}

auto CalcAccuracy(const std::vector<int> &predict,
                  const std::vector<int> &label) -> double {
    int correct = 0;
    for (int i = 0; i < predict.size(); i++) {
        if (predict[i] == label[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predict.size();
}
