#include "utils.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

int GLCM::kLevel = 16;
int GLCM::kD = 1;

auto GLCM::Compress(const cv::Mat &img) -> cv::Mat {
    cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    // img.convertTo(res, CV_8U, (level - 1) / 255.0); // Convert the grayscale
    // levels to level
    // *双重循环图像压缩，效率较低，但会发生四舍五入某些情况下分类准确率高于直接转换
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            res.at<uchar>(i, j) = img.at<uchar>(i, j) * (kLevel - 1) / 255;
        }
    }
    return res;
}

auto GLCM::GetGLCM(const cv::Mat &img, const double &angle) -> Eigen::MatrixXd {
    Eigen::MatrixXd glcm = Eigen::MatrixXd::Zero(kLevel, kLevel);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (round(j + kD * cos(angle)) < img.cols &&
                round(i + kD * sin(angle)) < img.rows &&
                round(j + kD * cos(angle)) >= 0 &&
                round(i + kD * sin(angle)) >= 0) {
                glcm(img.at<uchar>(i, j),
                     img.at<uchar>(
                         static_cast<int>(round(i + kD * sin(angle))),
                         static_cast<int>(round(j + kD * cos(angle)))))++;
            }
            if (round(j - kD * cos(angle)) < img.cols &&
                round(i - kD * sin(angle)) < img.rows &&
                round(j - kD * cos(angle)) >= 0 &&
                round(i - kD * sin(angle)) >= 0) {
                glcm(img.at<uchar>(i, j),
                     img.at<uchar>(
                         static_cast<int>(round(i - kD * sin(angle))),
                         static_cast<int>(round(j - kD * cos(angle)))))++;
            }
        }
    }
    glcm /= glcm.sum();
    return glcm;
}

auto GLCM::GetFeature(const Eigen::MatrixXd &glcm) -> Eigen::VectorXd {
    double energy = glcm.cwiseProduct(glcm).sum();
    double entropy = -glcm.cwiseProduct(glcm.unaryExpr([&](double x) {
                              return x == 0 ? 0 : log(x);
                          }))
                          .sum();
    double contrast = 0, homogeneity = 0;
    for (int i = 0; i < kLevel; i++) {
        for (int j = 0; j < kLevel; j++) {
            contrast += (i - j) * (i - j) * glcm(i, j);
            homogeneity += glcm(i, j) / (1 + (i - j) * (i - j));
        }
    }
    return Eigen::Vector4d({entropy, energy, contrast, homogeneity});
}

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

std::vector<double> Preprocess::kAngles = {0.0, M_PI / 4, M_PI / 2,
                                           M_PI * 3 / 4};

auto Preprocess::Preprocessing(const std::vector<cv::Mat> &data) const
    -> Eigen::MatrixXd {
    auto features = Eigen::MatrixXd();
    for (auto img : data) {
        auto feature = Eigen::MatrixXd(kAngles.size(), 0);
        for (auto angle : kAngles) {
            auto glcm = GLCM::GetGLCM(GLCM::Compress(img), angle);
            feature.conservativeResize(Eigen::NoChange, feature.cols() + 1);
            feature.col(feature.cols() - 1) = GLCM::GetFeature(glcm);
        }
        feature.resize(1, feature.size());
        features.conservativeResize(features.rows() + 1, feature.cols());
        features.row(features.rows() - 1) = feature;
    }
    if (strategy_ != nullptr) features = strategy_->Normalize(features);
    return features;
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
