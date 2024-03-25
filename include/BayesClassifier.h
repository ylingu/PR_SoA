#ifndef BAYESCLASSIFIER_H
#define BAYESCLASSIFIER_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

//灰度共生矩阵特征提取
class GLCM
{
private:
    static int level; //灰度级数
    static int d;     //距离
public:
    static auto compress(const cv::Mat &img) -> cv::Mat;
    static auto getGLCM(const cv::Mat &img, const double &angle) -> Eigen::MatrixXd;
    static auto getFeature(const Eigen::MatrixXd &GLCM) -> Eigen::VectorXd;
};

//贝叶斯分类器
class BayesClassifier
{
private:
    static std::vector<double> angles; //角度
    Eigen::MatrixXd mean;              //均值com
    Eigen::MatrixXd cov;               //协方差
    Eigen::VectorXd prior;             //先验概率
    Eigen::VectorXd max;               //最大值
    Eigen::VectorXd min;               //最小值
    Eigen::VectorXd meanZscore;        //使用z-score标准化时使用的均值
    Eigen::VectorXd stdZscore;         //使用z-score标准化时使用的标准差
public:
    BayesClassifier() {} //默认构造函数
    static auto Preprocessing(const std::vector<cv::Mat> &data) -> Eigen::MatrixXd;
    auto train(const std::vector<cv::Mat> &data, const std::vector<int> &label, const int &classNum)
        -> void;
    auto predict(const std::vector<cv::Mat> &data) const -> std::vector<int>;
    static auto calcAccuracy(const std::vector<int> &predict, const std::vector<int> &label)
        -> double;
};
#endif