#include "Utils.h"
#include <cmath>
#include <memory>

int GLCM::level = 256;
int GLCM::d = 1;

auto GLCM::Compress(const cv::Mat &img) -> cv::Mat
{
    cv::Mat res;
    img.convertTo(res, CV_8U, (level - 1) / 255.0); // Convert the grayscale levels to 16
    return res;
}

auto GLCM::GetGLCM(const cv::Mat &img, const double &angle) -> Eigen::MatrixXd
{
    Eigen::MatrixXd GLCM = Eigen::MatrixXd::Zero(level, level);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (round(j + d * cos(angle)) < img.cols && round(i + d * sin(angle)) < img.rows
                && round(j + d * cos(angle)) >= 0 && round(i + d * sin(angle)) >= 0) {
                GLCM(img.at<uchar>(i, j),
                     img.at<uchar>(static_cast<int>(round(i + d * sin(angle))),
                                   static_cast<int>(round(j + d * cos(angle)))))
                ++;
            }
            if (round(j - d * cos(angle)) < img.cols && round(i - d * sin(angle)) < img.rows
                && round(j - d * cos(angle)) >= 0 && round(i - d * sin(angle)) >= 0) {
                GLCM(img.at<uchar>(i, j),
                     img.at<uchar>(static_cast<int>(round(i - d * sin(angle))),
                                   static_cast<int>(round(j - d * cos(angle)))))
                ++;
            }
        }
    }
    GLCM /= GLCM.sum();
    return GLCM;
}

auto GLCM::GetFeature(const Eigen::MatrixXd &GLCM) -> Eigen::VectorXd
{
    double energy = GLCM.cwiseProduct(GLCM).sum();
    double entropy =
        -GLCM.cwiseProduct(GLCM.unaryExpr([&](double x) { return x == 0 ? 0 : log(x); })).sum();
    double contrast = 0, homogeneity = 0;
    for (int i = 0; i < level; i++) {
        for (int j = 0; j < level; j++) {
            contrast += (i - j) * (i - j) * GLCM(i, j);
            homogeneity += GLCM(i, j) / (1 + (i - j) * (i - j));
        }
    }
    return Eigen::Vector4d({entropy, energy, contrast, homogeneity});
}

auto LinearNormalization::Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd
{
    if (!parametersInitialized) {
        max = features.colwise().maxCoeff();
        min = features.colwise().minCoeff();
        parametersInitialized = true;
    }
    return (features.rowwise() - min.transpose()).array().rowwise()
           / (max - min).transpose().array();
}

auto ZScoreStandardization::Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd
{
    if (!parametersInitialized) {
        mean = features.colwise().mean();
        std = ((features.rowwise() - mean.transpose())
                   .cwiseProduct(features.rowwise() - mean.transpose()))
                  .colwise()
                  .sum()
                  .cwiseSqrt();
        parametersInitialized = true;
    }
    return (features.rowwise() - mean.transpose()).array().rowwise() / (std * 6).transpose().array()
           + 0.5;
}

std::vector<double> Preprocess::angles = {0.0, M_PI / 4, M_PI / 2, M_PI * 3 / 4};

auto Preprocess::Preprocessing(const std::vector<cv::Mat> &data) const -> Eigen::MatrixXd
{
    auto features = Eigen::MatrixXd();
    for (auto img : data) {
        auto feature = Eigen::MatrixXd(angles.size(), 0);
        for (auto angle : angles) {
            auto GLCM = GLCM::GetGLCM(GLCM::Compress(img), angle);
            feature.conservativeResize(Eigen::NoChange, feature.cols() + 1);
            feature.col(feature.cols() - 1) = GLCM::GetFeature(GLCM);
        }
        feature.resize(1, feature.size());
        features.conservativeResize(features.rows() + 1, feature.cols());
        features.row(features.rows() - 1) = feature;
    }
    if (strategy != nullptr)
        features = strategy->Normalize(features);
    return features;
}

auto calcAccuracy(const std::vector<int> &predict, const std::vector<int> &label) -> double
{
    int correct = 0;
    for (int i = 0; i < predict.size(); i++) {
        if (predict[i] == label[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / predict.size();
}
