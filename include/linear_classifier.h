// 确保头文件只被包含一次
#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <Eigen/Dense>

// 垂直平分线分类器
class VerticalBisectorClassifier {
private:
    Eigen::VectorXd w_, w_0_;  // 权重向量和偏置
public:
    VerticalBisectorClassifier(){};  // 默认构造函数
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);
    double Predict(const Eigen::VectorXd &x);
};

// 最小距离分类器
class MinimumDistanceClassifier {
private:
    Eigen::VectorXd m_1_, m_2_;  // 两个类别的均值
public:
    MinimumDistanceClassifier(){};  // 默认构造函数
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);
    bool Predict(const Eigen::VectorXd &x);
};

// Fisher投影准则
Eigen::VectorXd Fisher(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);

// 准则函数抽象类
class Criterion {
public:
    virtual double operator()(const Eigen::VectorXd &a,
                              const Eigen::MatrixXd &y);
    virtual double operator()(const Eigen::VectorXd &a,
                              const Eigen::MatrixXd &y,
                              const Eigen::VectorXd &b);
    virtual ~Criterion() = default;  // 虚析构函数
};

// 最小错分样本准则（与默认实现不同的另一种）
class MisclassificationCriterion : public Criterion {
public:
    double operator()(const Eigen::VectorXd &a,
                      const Eigen::MatrixXd &y) override;
};

// 最小平方误差准则
class LeastSquaresCriterion : public Criterion {
public:
    double operator()(const Eigen::VectorXd &a, const Eigen::MatrixXd &y,
                      const Eigen::VectorXd &b) override;
};

// 线性分类器
class LinearClassifier {
private:
    Eigen::VectorXd a_;  // 权重向量
    Criterion
        *c_;  // 准则函数，仅支持感知准则和最小平方误差准则(QAQ不会autograd)
public:
    LinearClassifier(Criterion *c) : c_(c) {}  // 构造函数
    Eigen::MatrixXd get_y(const Eigen::MatrixXd &x_1,
                          const Eigen::MatrixXd &x_2);
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2,
               const double &learning_rate, const int &epochs);
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2,
               const Eigen::VectorXd &b);
    double Predict(Eigen::VectorXd x);
};

#endif  // LINEARCLASSIFIER_H