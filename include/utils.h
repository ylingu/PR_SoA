#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// 灰度共生矩阵特征提取
class GLCM {
private:
    static int kLevel;  // 灰度级数
    static int kD;      // 距离
public:
    static auto Compress(const cv::Mat &img) -> cv::Mat;
    static auto GetGLCM(const cv::Mat &img, const double &angle)
        -> Eigen::MatrixXd;
    static auto GetFeature(const Eigen::MatrixXd &glcm) -> Eigen::VectorXd;
};

class Normalization {
protected:
    bool parametersInitialized_ = false;

public:
    virtual auto Normalize(const Eigen::MatrixXd &features)
        -> Eigen::MatrixXd = 0;
    virtual ~Normalization() = default;
};

class LinearNormalization : public Normalization {
private:
    Eigen::VectorXd max_;  // 最大值
    Eigen::VectorXd min_;  // 最小值
public:
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

class ZScoreStandardization : public Normalization {
private:
    Eigen::VectorXd mean_;  // 均值
    Eigen::VectorXd std_;   // 标准差
public:
    auto Normalize(const Eigen::MatrixXd &features) -> Eigen::MatrixXd override;
};

class Preprocess {
private:
    static std::vector<double> kAngles;        // 角度
    std::unique_ptr<Normalization> strategy_;  // 归一化策略
public:
    Preprocess(std::unique_ptr<Normalization> strategy)
        : strategy_(std::move(strategy)) {}
    auto Preprocessing(const std::vector<cv::Mat> &data) const
        -> Eigen::MatrixXd;
};

auto CalcAccuracy(const std::vector<int> &predict,
                  const std::vector<int> &label) -> double;

#endif