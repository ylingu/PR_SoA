#include <opencv2/core/hal/interface.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "feature_extraction.h"

int FeatureExtraction::kLevel = 16;
auto FeatureExtraction::Compress(const cv::Mat &img) -> cv::Mat {
    cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    img.convertTo(res, CV_8U, (kLevel - 1) / 255.0);
    // *双重循环图像压缩，效率较低，但会发生四舍五入某些情况下分类准确率高于直接转换
    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //         res.at<uchar>(i, j) = img.at<uchar>(i, j) * (kLevel - 1) / 255;
    //     }
    // }
    return res;
}

auto GLCMFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    Eigen::VectorXd feature = Eigen::VectorXd::Zero(angles_.size() * 4);
    for (auto it = angles_.begin(); it != angles_.end(); it++) {
        Eigen::MatrixXd glcm = Eigen::MatrixXd::Zero(kLevel, kLevel);
        for (int i = 0; i != img.rows; ++i) {
            for (int j = 0; j != img.cols; ++j) {
                if (round(j + d_ * cos(*it)) < img.cols &&
                    round(i + d_ * sin(*it)) < img.rows &&
                    round(j + d_ * cos(*it)) >= 0 &&
                    round(i + d_ * sin(*it)) >= 0) {
                    glcm(img.at<uchar>(i, j),
                         img.at<uchar>(
                             static_cast<int>(round(i + d_ * sin(*it))),
                             static_cast<int>(round(j + d_ * cos(*it)))))++;
                }
                if (round(j - d_ * cos(*it)) < img.cols &&
                    round(i - d_ * sin(*it)) < img.rows &&
                    round(j - d_ * cos(*it)) >= 0 &&
                    round(i - d_ * sin(*it)) >= 0) {
                    glcm(img.at<uchar>(i, j),
                         img.at<uchar>(
                             static_cast<int>(round(i - d_ * sin(*it))),
                             static_cast<int>(round(j - d_ * cos(*it)))))++;
                }
            }
        }
        glcm /= glcm.sum();
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
        feature.segment(std::distance(angles_.begin(), it) * 4, 4) =
            Eigen::Vector4d(entropy, energy, contrast, homogeneity);
    }
    return feature;
}

auto GLCMFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    Eigen::MatrixXd features =
        Eigen::MatrixXd::Zero(data.size(), angles_.size() * 4);
    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        features.row(std::distance(data.begin(), it)) = feature;
    }
    return features;
}

auto GaborFeatureExtraction::Extract(const cv::Mat &img) -> Eigen::VectorXd {
    Eigen::VectorXd feature = Eigen::VectorXd::Zero(kernels_.size() * 4);
    cv::Mat dest;
    for (auto it = kernels_.begin(); it != kernels_.end(); ++it) {
        cv::filter2D(img, dest, CV_32F, *it);
        cv::Scalar mean, stddev;
        cv::meanStdDev(dest, mean, stddev);
        cv::Mat squared;
        cv::multiply(dest, dest, squared);
        double energy = cv::sum(squared)[0];
        cv::Mat hist;
        float range[] = {0, static_cast<float>(kLevel)};
        const float *hist_range[] = {range};
        cv::calcHist(
            &dest, 1, 0, cv::Mat(), hist, 1, &kLevel, hist_range, true, false);
        hist /= dest.total();
        cv::Mat log_p;
        cv::log(hist, log_p);
        double entropy = -cv::sum(hist.mul(log_p))[0];
        feature.segment(std::distance(kernels_.begin(), it) * 4, 4) =
            Eigen::Vector4d(mean[0], stddev[0], energy, entropy);
    }
    return feature;
}

auto GaborFeatureExtraction::BatchExtract(const std::vector<cv::Mat> &data)
    -> Eigen::MatrixXd {
    if (kernels_.empty()) {
        GetGaborKernel();
    }
    Eigen::MatrixXd features =
        Eigen::MatrixXd::Zero(data.size(), kernels_.size() * 4);
    for (auto it = data.begin(); it != data.end(); it++) {
        auto img = Compress(*it);
        auto feature = Extract(img);
        features.row(std::distance(data.begin(), it)) = feature;
    }
    return features;
}