#include "LinearClassifier.h"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

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
//默认实现，使用感知准则
double Criterion::operator()(const Eigen::VectorXd &a, const Eigen::MatrixXd &Y)
{
    return (-Y * a).cwiseMax(0).sum();
}
//默认实现，使用一种最小错分样本准则
double Criterion::operator()(const Eigen::VectorXd &a, const Eigen::MatrixXd &Y,
                             const Eigen::VectorXd &b)
{
    Eigen::VectorXd X = (Y * a - b) - (Y * a - b).cwiseAbs();
    return X.squaredNorm();
}

double MisclassificationCriterion::operator()(const Eigen::VectorXd &a, const Eigen::MatrixXd &Y)
{
    return (Y * a).unaryExpr([](const double x) { return x >= 0 ? 1 : 0; }).sum();
}

double LeastSquaresCriterion::operator()(const Eigen::VectorXd &a, const Eigen::MatrixXd &Y,
                                         const Eigen::VectorXd &b)
{
    return (Y * a - b).squaredNorm();
}

Eigen::MatrixXd LinearClassifier::getY(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2)
{
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(X_1.cols(), 1);
    Eigen::MatrixXd X1(X_1.cols(), X_1.rows() + 1);
    X1 << X_1.transpose(), ones;
    ones = Eigen::MatrixXd::Ones(X_2.cols(), 1);
    Eigen::MatrixXd X2(X_2.cols(), X_2.rows() + 1);
    X2 << X_2.transpose(), ones;
    Eigen::MatrixXd Y(X1.rows() + X2.rows(), X1.cols());
    Y << X1, -X2;
    return Y;
}

void LinearClassifier::train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2,
                             const Eigen::VectorXd &b)
{
    Eigen::MatrixXd Y = getY(X_1, X_2);
    double epsilon = std::numeric_limits<double>::epsilon();
    Eigen::MatrixXd temp = Y.transpose() * Y;
    //计算广义逆，使用SVD分解(Copilot写的，我也不知道对不对)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(temp, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance =
        epsilon * std::max(temp.cols(), temp.rows()) * svd.singularValues().array().abs()(0);
    temp = svd.matrixV()
           * (svd.singularValues().array().abs() > tolerance)
                 .select(svd.singularValues().array().inverse(), 0)
                 .matrix()
                 .asDiagonal()
           * svd.matrixU().adjoint();
    a = temp * Y.transpose() * b;
    a = (Y.transpose() * Y).inverse() * Y.transpose() * b;
}

void LinearClassifier::train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2,
                             const double &lr, const int &epochs)
{
    Eigen::MatrixXd Y = getY(X_1, X_2);
    a = Eigen::VectorXd::Zero(Y.cols());
    for (int i = 0; i < epochs; i++) {
        Eigen::MatrixXd temp =
            (-Y * a).unaryExpr([&](double val) { return val >= 0 ? 1.0 : 0; }).asDiagonal() * Y;
        Eigen::VectorXd X = temp.colwise().sum();
        a += lr * X;
        //调用准则函数
        double loss = (*c)(a, Y);
        std::cout << "Epoch " << i + 1 << " Loss: " << loss << std::endl;
    }
}

double LinearClassifier::predict(Eigen::VectorXd x)
{
    x.conservativeResize(x.size() + 1);
    x(x.size() - 1) = 1;
    return (a.transpose() * x).value();
}