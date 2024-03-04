// 确保头文件只被包含一次
#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <Eigen/Dense>
#include <iostream>

//垂直平分线分类器
class VerticalBisectorClassifier
{
private:
    Eigen::VectorXd w, w_0; //权重向量和偏置
public:
    VerticalBisectorClassifier(); //默认构造函数
    void train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2);
    double predict(const Eigen::VectorXd &x);
};

//最小距离分类器
class MinimumDistanceClassifier
{
private:
    Eigen::VectorXd m_1, m_2; //两个类别的均值
public:
    MinimumDistanceClassifier(); //默认构造函数
    void train(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2);
    bool predict(const Eigen::VectorXd &x);
};

//Fisher投影
Eigen::VectorXd fisher(const Eigen::MatrixXd &X_1, const Eigen::MatrixXd &X_2);
#endif // LINEARCLASSIFIER_H