#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <print>
#include <string>
#include <vector>

#include "feature_extraction.h"
#include "unsupervised_learning.h"

int main() {
    std::vector<cv::Mat> imgs;
    for (int i = 1; i != 21; ++i) {
        for (int j = 1; j != 21; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            imgs.push_back(img);
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
    KmeansClustering clustering(
        std::make_unique<GaborFeatureExtraction>(thetas, lambdas),
        20,
        cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, 0.5));
    // DbscanClustering clustering(
    //     std::make_unique<GLCMFeatureExtraction>(thetas, 1),
    //     0.2,
    //     5);
    auto data = clustering.Preprocess(imgs, 30);
    auto labels = clustering.Cluster(data);
    for (int i = 0; i != 20; ++i) {
        for (int j = 0; j != 20; ++j) {
            std::print("{} ", labels[i * 20 + j]);
        }
        std::print("\n");
    }
    return 0;
}