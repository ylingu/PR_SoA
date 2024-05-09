#include "linear_classifier.h"

#include <Eigen/Dense>
#include <print>
#include <stdexcept>

void VerticalBisectorClassifier::Train(const Eigen::MatrixXd &x_1,
                                       const Eigen::MatrixXd &x_2) {
    // 计算两个类别的均值
    Eigen::VectorXd m_1 = x_1.rowwise().mean();
    Eigen::VectorXd m_2 = x_2.rowwise().mean();
    // 计算权重向量
    w_ = m_1 - m_2;
    w_0_ = 0.5 * w_.transpose() * (m_1 + m_2);
}
double VerticalBisectorClassifier::Predict(const Eigen::VectorXd &x) {
    return (w_.transpose() * x - w_0_).value();
}

void MinimumDistanceClassifier::Train(const Eigen::MatrixXd &x_1,
                                      const Eigen::MatrixXd &x_2) {
    // 计算两个类别的均值
    m_1_ = x_1.rowwise().mean();
    m_2_ = x_2.rowwise().mean();
}
bool MinimumDistanceClassifier::Predict(const Eigen::VectorXd &x) {
    return (m_1_ - x).squaredNorm() < (m_2_ - x).squaredNorm();
}

Eigen::VectorXd Fisher(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2) {
    // 计算两个类别的均值
    Eigen::VectorXd m_1 = x_1.rowwise().mean();
    Eigen::VectorXd m_2 = x_2.rowwise().mean();
    // 计算类内散度矩阵
    Eigen::MatrixXd s_1 =
        (x_1.colwise() - m_1) * (x_1.colwise() - m_1).transpose();
    Eigen::MatrixXd s_2 =
        (x_2.colwise() - m_2) * (x_2.colwise() - m_2).transpose();
    // 计算类间散度矩阵
    Eigen::MatrixXd s_w = s_1 + s_2;
    // 检查S_w是否可逆
    if (s_w.determinant() == 0) {
        throw std::runtime_error("S_w is singular!");  // 抛出运行时错误异常
    }
    return s_w.inverse() * (m_1 - m_2);
}
// 默认实现，使用感知准则
double Criterion::operator()(const Eigen::VectorXd &a,
                             const Eigen::MatrixXd &y) {
    return (-y * a).cwiseMax(0).sum();
}
// 默认实现，使用一种最小错分样本准则
double Criterion::operator()(const Eigen::VectorXd &a,
                             const Eigen::MatrixXd &y,
                             const Eigen::VectorXd &b) {
    Eigen::VectorXd x = (y * a - b) - (y * a - b).cwiseAbs();
    return x.squaredNorm();
}

double MisclassificationCriterion::operator()(const Eigen::VectorXd &a,
                                              const Eigen::MatrixXd &y) {
    return (y * a)
        .unaryExpr([](const double x) { return x >= 0 ? 1 : 0; })
        .sum();
}

double LeastSquaresCriterion::operator()(const Eigen::VectorXd &a,
                                         const Eigen::MatrixXd &y,
                                         const Eigen::VectorXd &b) {
    return (y * a - b).squaredNorm();
}

Eigen::MatrixXd LinearClassifier::get_y(const Eigen::MatrixXd &x_1,
                                        const Eigen::MatrixXd &x_2) {
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(x_1.cols(), 1);
    Eigen::MatrixXd x1(x_1.cols(), x_1.rows() + 1);
    x1 << x_1.transpose(), ones;
    ones = Eigen::MatrixXd::Ones(x_2.cols(), 1);
    Eigen::MatrixXd x2(x_2.cols(), x_2.rows() + 1);
    x2 << x_2.transpose(), ones;
    Eigen::MatrixXd y(x1.rows() + x2.rows(), x1.cols());
    y << x1, -x2;
    return y;
}

void LinearClassifier::Train(const Eigen::MatrixXd &x_1,
                             const Eigen::MatrixXd &x_2,
                             const Eigen::VectorXd &b) {
    Eigen::MatrixXd y = get_y(x_1, x_2);
    double epsilon = std::numeric_limits<double>::epsilon();
    Eigen::MatrixXd temp = y.transpose() * y;
    // 计算广义逆，使用SVD分解(Copilot写的，我也不知道对不对)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        temp, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(temp.cols(), temp.rows()) *
                       svd.singularValues().array().abs()(0);
    temp = svd.matrixV() *
           (svd.singularValues().array().abs() > tolerance)
               .select(svd.singularValues().array().inverse(), 0)
               .matrix()
               .asDiagonal() *
           svd.matrixU().adjoint();
    a_ = temp * y.transpose() * b;
    a_ = (y.transpose() * y).inverse() * y.transpose() * b;
}

void LinearClassifier::Train(const Eigen::MatrixXd &x_1,
                             const Eigen::MatrixXd &x_2,
                             const double &learning_rate,
                             const int &epochs) {
    Eigen::MatrixXd y = get_y(x_1, x_2);
    a_ = Eigen::VectorXd::Zero(y.cols());
    for (int i = 0; i < epochs; i++) {
        Eigen::MatrixXd temp =
            (-y * a_)
                .unaryExpr([&](double val) { return val >= 0 ? 1.0 : 0; })
                .asDiagonal() *
            y;
        Eigen::VectorXd x = temp.colwise().sum();
        a_ += learning_rate * x;
        // 调用准则函数
        double loss = (*c_)(a_, y);
        std::print("Epoch: {} Loss: {}\n",i + 1, loss);
    }
}

double LinearClassifier::Predict(Eigen::VectorXd x) {
    x.conservativeResize(x.size() + 1);
    x(x.size() - 1) = 1;
    return (a_.transpose() * x).value();
}