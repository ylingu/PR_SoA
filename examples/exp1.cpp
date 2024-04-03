#include "BayesClassifier.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> trainData, testData;
    std::vector<int> trainLabel, testLabel;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 17; ++j) {
            auto img = cv::imread("../../../../examples/imgs/s" + std::to_string(i) + "/"
                                      + std::to_string(j) + ".bmp",
                                  cv::IMREAD_GRAYSCALE);
            trainData.push_back(img);
            trainLabel.push_back(i - 1);
        }
        for (; j < 21; ++j) {
            auto img = cv::imread("../../../../examples/imgs/s" + std::to_string(i) + "/"
                                      + std::to_string(j) + ".bmp",
                                  cv::IMREAD_GRAYSCALE);
            testData.push_back(img);
            testLabel.push_back(i - 1);
        }
    }
    BayesClassifier classifier(std::make_unique<LinearNormalization>());
    classifier.train(trainData, trainLabel, 20);
    auto predict = classifier.predict(testData);
    std::cout << "Accuracy: " << calcAccuracy(predict, testLabel) << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;
    return 0;
}