#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "bayes_classifier.h"
#include "utils.h"
int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> train_data, test_data;
    std::vector<int> train_label, test_label;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 17; ++j) {
            auto img =
                cv::imread("../../../../examples/imgs/s" + std::to_string(i) +
                               "/" + std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            train_data.push_back(img);
            train_label.push_back(i - 1);
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
    BayesClassifier classifier(std::make_unique<LinearNormalization>());
    classifier.Train(train_data, train_label, 20);
    auto predict = classifier.Predict(test_data);
    std::cout << "Accuracy: " << CalcAccuracy(predict, test_label) << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    return 0;
}