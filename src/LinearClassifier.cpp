#include "LinearClassifier.h"

VerticalBisectorClassifier::VerticalBisectorClassifier(){}; //默认构造函数
void VerticalBisectorClassifier::train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2)
{
    //计算两个类别的均值
    Eigen::VectorXd m_1 = X_1.rowwise().mean();
    Eigen::VectorXd m_2 = X_2.rowwise().mean();
    //计算权重向量
    w = m_1 - m_2;
    w_0 = 0.5 * w.transpose() * (m_1 + m_2);
}
double VerticalBisectorClassifier::predict(const Eigen::VectorXd &x)
{
    return (w.transpose() * x - w_0).value();
}

MinimumDistanceClassifier::MinimumDistanceClassifier(){}; //默认构造函数
void MinimumDistanceClassifier::train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2)
{
    //计算两个类别的均值
    m_1 = X_1.rowwise().mean();
    m_2 = X_2.rowwise().mean();
}
bool MinimumDistanceClassifier::predict(const Eigen::VectorXd &x)
{
    return (m_1 - x).squaredNorm() < (m_2 - x).squaredNorm();
}

Eigen::VectorXd fisher(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2)
{
    //计算两个类别的均值
    Eigen::VectorXd m_1 = X_1.rowwise().mean();
    Eigen::VectorXd m_2 = X_2.rowwise().mean();
    //计算类内散度矩阵
    Eigen::MatrixXd S_1 = (X_1.colwise() - m_1) * (X_1.colwise() - m_1).transpose();
    Eigen::MatrixXd S_2 = (X_2.colwise() - m_2) * (X_2.colwise() - m_2).transpose();
    //计算类间散度矩阵
    Eigen::MatrixXd S_w = S_1 + S_2;
    //检查S_w是否可逆
    if (S_w.determinant() == 0) {
        std::cerr << "S_w is singular!" << std::endl;
        return Eigen::VectorXd();
    }
    return S_w.inverse() * (m_1 - m_2);
}