#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include "Utils.h"

//贝叶斯分类器
class BayesClassifier
{
private:
    Eigen::MatrixXd mean;  //均值
    Eigen::MatrixXd cov;   //协方差
    Eigen::VectorXd prior; //先验概率
    Preprocess Preprocessor;

public:
    BayesClassifier(std::unique_ptr<Normalization> strategy) : Preprocessor(std::move(strategy)) {}
    auto train(const std::vector<cv::Mat> &data, const std::vector<int> &label, const int &classNum)
        -> void;
    auto predict(const std::vector<cv::Mat> &data) const -> std::vector<int>;
};
#endif