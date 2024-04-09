#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "nonlinear_classifier.h"
#include "utils.h"

int main() {
    std::vector<cv::Mat> train_data, test_data, validate_data;
    std::vector<int> train_label, test_label, validate_label;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 13; ++j) {
            auto img =
                cv::imread("../../../../examples/imgs/s" + std::to_string(i) +
                               "/" + std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            train_data.push_back(img);
            train_label.push_back(i - 1);
        }
        for (; j < 17; ++j) {
            auto img =
                cv::imread("../../../../examples/imgs/s" + std::to_string(i) +
                               "/" + std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            validate_data.push_back(img);
            validate_label.push_back(i - 1);
        }
        for (; j < 21; ++j) {
            auto img =
                cv::imread("../../../../examples/imgs/s" + std::to_string(i) +
                               "/" + std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            test_data.push_back(img);
            test_label.push_back(i - 1);
        }
    }
    SVMClassifier classifier(std::make_unique<LinearNormalization>());
    std::vector<double> c_range, gamma_range;
    for (double i = 19; i < 21; i += 0.01) {
        c_range.push_back(i);
    }
    for (double i = 4; i < 6; i += 0.01) {
        gamma_range.push_back(i);
    }
    classifier.Train(train_data, train_label, validate_data, validate_label,
                     c_range, gamma_range, 20);
    auto test_features64 = classifier.preprocessor_.Preprocessing(test_data);
    Eigen::MatrixXf test_features = test_features64.cast<float>();
    cv::Mat test_data_mat;
    cv::eigen2cv(test_features, test_data_mat);
    auto predict = classifier.Predict(test_data_mat);
    classifier.SaveModel();
    std::cout << "Accuracy: " << CalcAccuracy(predict, test_label) << std::endl;
    return 0;
}