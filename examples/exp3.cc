#include <memory>
#include <opencv2/opencv.hpp>
#include <print>
#include <string>
#include <vector>

#include "bayes_classifier.h"
#include "csv.h"
#include "feature_extraction.h"
#include "utils.h"

int main() {
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
    std::vector<double> thetas = {0,
                                  CV_PI / 8,
                                  CV_PI / 4,
                                  3 * CV_PI / 8,
                                  CV_PI / 2,
                                  5 * CV_PI / 8,
                                  3 * CV_PI / 4,
                                  7 * CV_PI / 8};
    std::vector<double> lambdas = {2, pow(2, 1.5), 4, pow(2, 2.5), 8};
    auto train_data = Eigen::MatrixXd(), test_data = Eigen::MatrixXd();
    std::vector<std::vector<std::string>> data{{"pca_dims", "accuracy"}};
    CSV csv(data);
    for (int i = 1; i != 161; ++i) {
        BayesClassifier classifier(
            20,
            std::make_unique<LinearNormalization>(),
            std::make_unique<GaborFeatureExtraction>(thetas, lambdas));
        train_data = classifier.Preprocess(train_imgs, i);
        test_data = classifier.Preprocess(test_imgs, i);
        classifier.Train(train_data, train_label);
        auto predict = classifier.Predict(test_data);
        auto accuracy = CalcAccuracy(predict, test_label);
        csv.InsertRow(csv.GetRowCount(),
                      {std::to_string(i), std::to_string(accuracy * 100)});
        std::print("PCA dims: {}, Accuracy: {:.2f}%\n", i, accuracy * 100);
    }
    csv.Save("exp3.csv");
    return 0;
}