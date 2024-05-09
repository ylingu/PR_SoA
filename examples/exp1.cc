#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <print>
#include <string>
#include <vector>

#include "bayes_classifier.h"
#include "utils.h"

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> train_imgs, test_imgs;
    std::vector<int> train_label, test_label;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 17; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            train_imgs.push_back(img);
            train_label.push_back(i - 1);
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
    BayesClassifier classifier(20, std::make_unique<LinearNormalization>());
    auto train_data = classifier.Preprocess(train_imgs),
         test_data = classifier.Preprocess(test_imgs);
    classifier.Train(train_data, train_label);
    auto predict = classifier.Predict(test_data);
    std::print("Accuracy: {}\n", CalcAccuracy(predict, test_label));
    auto end = std::chrono::high_resolution_clock::now();
    std::print(
        "Time: {}ms\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());
    return 0;
}