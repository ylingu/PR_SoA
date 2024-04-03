#include "BayesClassifier.h"

auto BayesClassifier::train(const std::vector<cv::Mat> &data, const std::vector<int> &label,
                            const int &classNum) -> void
{
    int n = data.size();
    auto features = Preprocessor.Preprocessing(data);
    // * Calculate the mean, covariance, and prior probability of each class
    int featureNum = features.row(0).size();
    mean = Eigen::MatrixXd::Zero(classNum, featureNum);
    cov = Eigen::MatrixXd::Zero(classNum, featureNum);
    prior = Eigen::VectorXd::Zero(classNum);
    for (int i = 0; i < n; ++i) {
        mean.row(label[i]) += features.row(i);
        prior(label[i])++;
    }
    for (int i = 0; i < classNum; ++i) {
        mean.row(i) /= prior(i);
    }
    for (int i = 0; i < n; ++i) {
        cov.row(label[i]) += (features.row(i) - mean.row(label[i]))
                                 .cwiseProduct(features.row(i) - mean.row(label[i]));
    }
    for (int i = 0; i < classNum; ++i) {
        cov.row(i) /= prior(i);
    }
    prior /= n;
}

auto BayesClassifier::predict(const std::vector<cv::Mat> &data) const -> std::vector<int>
{
    int n = data.size();
    auto features = Preprocessor.Preprocessing(data);
    auto res = std::vector<int>();
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(mean.rows());
        for (int j = 0; j < mean.rows(); ++j) {
            prob(j) = -0.5
                          * (features.row(i) - mean.row(j))
                                .cwiseProduct(cov.row(j).cwiseInverse())
                                .cwiseProduct(features.row(i) - mean.row(j))
                                .sum()
                      - 0.5 * log(cov.row(j).prod()) + log(prior(j));
        }
        int maxIndex;
        prob.maxCoeff(&maxIndex);
        res.push_back(maxIndex);
    }
    return res;
}
