#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

//灰度共生矩阵特征提取
class GLCM
{
private:
    static int level; //灰度级数
    static int d;     //距离
public:
    static auto Compress(const cv::Mat &img) -> cv::Mat;
    static auto GetGLCM(const cv::Mat &img, const double &angle) -> Eigen::MatrixXd;
    static auto GetFeature(const Eigen::MatrixXd &GLCM) -> Eigen::VectorXd;
};

class Normalization
{
protected:
    bool parametersInitialized = false;

public:
    virtual auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd = 0;
    virtual ~Normalization() = default;
};

class LinearNormalization : public Normalization
{
private:
    Eigen::VectorXd max; //最大值
    Eigen::VectorXd min; //最小值
public:
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

class ZScoreStandardization : public Normalization
{
private:
    Eigen::VectorXd mean; //均值
    Eigen::VectorXd std;  //标准差
public:
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

class Preprocess
{
private:
    static std::vector<double> angles;       //角度
    std::unique_ptr<Normalization> strategy; //归一化策略
public:
    Preprocess(std::unique_ptr<Normalization> strategy = nullptr) : strategy(std::move(strategy)) {}
    auto Preprocessing(const std::vector<cv::Mat> &data) const -> Eigen::MatrixXd;
};

auto calcAccuracy(const std::vector<int> &predict, const std::vector<int> &label) -> double;

#endif