#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "nonlinear_classifier.h"
#include "utils.h"

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> train_imgs, test_imgs, validate_imgs;
    std::vector<int> train_label, test_label, validate_label;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 13; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            train_imgs.push_back(img);
            train_label.push_back(i - 1);
        }
        for (; j < 17; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            validate_imgs.push_back(img);
            validate_label.push_back(i - 1);
        }
        for (; j < 21; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            test_imgs.push_back(img);
            test_label.push_back(i - 1);
        }
    }
    SVMClassifier classifier(20, std::make_unique<LinearNormalization>());
    std::vector<double> c_range, gamma_range;
    for (double i = 19; i < 21; i += 0.01) {
        c_range.push_back(i);
    }
    for (double i = 4; i < 6; i += 0.01) {
        gamma_range.push_back(i);
    }
    auto train_data = classifier.Preprocess(train_imgs),
         validate_data = classifier.Preprocess(validate_imgs),
         test_data = classifier.Preprocess(test_imgs);
    classifier.Train(train_data,
                     train_label,
                     validate_data,
                     validate_label,
                     c_range,
                     gamma_range);
    auto predict = classifier.Predict(test_data);
    classifier.SaveModel();
    std::cout << "Accuracy: " << CalcAccuracy(predict, test_label) << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    return 0;
}