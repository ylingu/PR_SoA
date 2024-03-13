#include "BayesClassifier.h"

int GLCM::level = 16;
int GLCM::d = 1;

auto GLCM::compress(const cv::Mat &img) -> cv::Mat
{
    cv::Mat res;
    img.convertTo(res, CV_8U, (level - 1) / 255.0); // Convert the grayscale levels to 16
    return res;
}

auto GLCM::getGLCM(const cv::Mat &img, const double &angle) -> Eigen::MatrixXd
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

auto GLCM::getFeature(const Eigen::MatrixXd &GLCM) -> Eigen::VectorXd
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

std::vector<double> BayesClassifier::angles = {0.0, M_PI / 4, M_PI / 2, M_PI * 3 / 4};

auto BayesClassifier::Preprocessing(const std::vector<cv::Mat> &data) -> Eigen::MatrixXd
{
    auto features = Eigen::MatrixXd();
    for (auto img : data) {
        auto feature = Eigen::MatrixXd(angles.size(), 0);
        for (auto angle : angles) {
            auto GLCM = GLCM::getGLCM(GLCM::compress(img), angle);
            feature.conservativeResize(Eigen::NoChange, feature.cols() + 1);
            feature.col(feature.cols() - 1) = GLCM::getFeature(GLCM);
        }
        feature.resize(1, feature.size());
        features.conservativeResize(features.rows() + 1, feature.cols());
        features.row(features.rows() - 1) = feature;
    }
    return features;
}

auto BayesClassifier::train(const std::vector<cv::Mat> &data, const std::vector<int> &label,
                            const int &classNum) -> void
{
    int n = data.size();
    auto features = Preprocessing(data);
    Eigen::VectorXd feature;
    // * Linear standardization
    max = features.row(0);
    min = max;
    for (int i = 0; i < n; ++i) {
        feature = features.row(i);
        max = max.cwiseMax(feature);
        min = min.cwiseMin(feature);
    }
    for (int i = 0; i < n; ++i) {
        feature = features.row(i);
        features.row(i) = (feature - min).cwiseQuotient(max - min);
    }
    // * Z-score standardization
    // meanZscore = features.colwise().mean();
    // stdZscore = ((features.rowwise() - meanZscore.transpose())
    //                  .cwiseProduct(features.rowwise() - meanZscore.transpose()))
    //                 .colwise()
    //                 .sum()
    //                 .cwiseSqrt();
    // for (int i = 0; i < features.cols(); ++i) {
    //     feature = features.row(i);
    //     features.row(i) = (feature - meanZscore).cwiseQuotient(6 * stdZscore).array() + 0.5;
    // }
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

auto BayesClassifier::predict(const std::vector<cv::Mat> &data) -> std::vector<int>
{
    int n = data.size();
    auto features = Preprocessing(data);
    Eigen::VectorXd feature;
    // * Linear standardization
    for (int i = 0; i < n; ++i) {
        feature = features.row(i);
        features.row(i) = (feature - min).cwiseQuotient(max - min);
    }
    // * Z-score standardization
    // for (int i = 0; i < features.cols(); ++i) {
    //     feature = features.row(i);
    //     features.row(i) = (feature - meanZscore).cwiseQuotient(6 * stdZscore).array() + 0.5;
    // }
    auto res = std::vector<int>();
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd prob = Eigen::VectorXd::Zero(mean.rows());
        for (int j = 0; j < mean.rows(); ++j) {
            prob(j) = prior(j)
                      * (1.0 / pow(2 * M_PI, mean.cols() / 2) / pow(cov.row(j).prod(), 0.5))
                      * exp(-0.5
                            * (features.row(i) - mean.row(j))
                                  .cwiseProduct(cov.row(j).cwiseInverse())
                                  .cwiseProduct(features.row(i) - mean.row(j))
                                  .sum());
        }
        int maxIndex;
        prob.maxCoeff(&maxIndex);
        res.push_back(maxIndex);
    }
    return res;
}
