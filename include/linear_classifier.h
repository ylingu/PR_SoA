#ifndef LINEARCLASSIFIER_H
#define LINEARCLASSIFIER_H

#include <Eigen/Dense>

/**
 * @class VerticalBisectorClassifier
 * @brief A classifier that uses the vertical bisector method.
 */
class VerticalBisectorClassifier {
private:
    Eigen::VectorXd w_;    ///< Weight vector
    Eigen::VectorXd w_0_;  ///< Bias vector

public:
    /**
     * @brief Default constructor
     */
    VerticalBisectorClassifier(){};

    /**
     * @brief Train the model
     * @param x_1 Samples from the first class
     * @param x_2 Samples from the second class
     */
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);

    /**
     * @brief Predict the class of a sample
     * @param x The sample to predict
     * @return The predicted class
     */
    double Predict(const Eigen::VectorXd &x);
};

/**
 * @class MinimumDistanceClassifier
 * @brief A classifier that uses the minimum distance method.
 */
class MinimumDistanceClassifier {
private:
    Eigen::VectorXd m_1_;  ///< Mean vector of the first class
    Eigen::VectorXd m_2_;  ///< Mean vector of the second class

public:
    /**
     * @brief Default constructor
     */
    MinimumDistanceClassifier(){};

    /**
     * @brief Train the model
     * @param x_1 Samples from the first class
     * @param x_2 Samples from the second class
     */
    void Train(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);

    /**
     * @brief Predict the class of a sample
     * @param x The sample to predict
     * @return The predicted class
     */
    bool Predict(const Eigen::VectorXd &x);
};

/**
 * @brief Fisher projection criterion function.
 * @param x_1 First class samples.
 * @param x_2 Second class samples.
 * @return Fisher vector.
 */
Eigen::VectorXd Fisher(const Eigen::MatrixXd &x_1, const Eigen::MatrixXd &x_2);

/**
 * @class Criterion
 * @brief Abstract class for criterion functions.
 */
class Criterion {
public:
    /**
     * @brief Operator overloading for criterion function
     * @param a The vector to be projected
     * @param y The matrix of samples
     * @return The criterion value
     */
    virtual double operator()(const Eigen::VectorXd &a,
                              const Eigen::MatrixXd &y);

    /**
     * @brief Operator overloading for criterion function with bias
     * @param a The vector to be projected
     * @param y The matrix of samples
     * @param b The bias vector
     * @return The criterion value
     */
    virtual double operator()(const Eigen::VectorXd &a,
                              const Eigen::MatrixXd &y,
                              const Eigen::VectorXd &b);

    /**
     * @brief Default destructor
     */
    virtual ~Criterion() = default;
};

/**
 * @class MisclassificationCriterion
 * @brief Criterion function for misclassification.
 */
class MisclassificationCriterion : public Criterion {
public:
    /**
     * @brief Operator overloading for misclassification criterion function
     * @param a The vector to be projected
     * @param y The matrix of samples
     * @return The misclassification criterion value
     */
    double operator()(const Eigen::VectorXd &a,
                      const Eigen::MatrixXd &y) override;
};

/**
 * @class LeastSquaresCriterion
 * @brief Criterion function for least squares.
 */
class LeastSquaresCriterion : public Criterion {
public:
    /**
     * @brief Operator overloading for least squares criterion function
     * @param a The vector to be projected
     * @param y The matrix of samples
     * @param b The bias vector
     * @return The least squares criterion value
     */
    double operator()(const Eigen::VectorXd &a,
                      const Eigen::MatrixXd &y,
                      const Eigen::VectorXd &b) override;
};

/**
 * @class LinearClassifier
 * @brief A linear classifier.
 */
class LinearClassifier {
private:
    Eigen::VectorXd a_;  ///< The vector to be projected
    Criterion *c_;       ///< The criterion function pointer

public:
    /**
     * @brief Constructor that accepts a criterion function pointer
     * @param c The criterion function pointer
     */
    LinearClassifier(Criterion *c) : c_(c) {}

    /**
     * @brief Get the labels of the samples
     * @param x_1 Samples from the first class
     * @param x_2 Samples from the second class
     * @return The labels of the samples
     */
    Eigen::MatrixXd get_y(const Eigen::MatrixXd &x_1,
                          const Eigen::MatrixXd &x_2);

    /**
     * @brief Train the model
     * @param x_1 Samples from the first class
     * @param x_2 Samples from the second class
     * @param learning_rate The learning rate
     * @param epochs The number of epochs
     */
    void Train(const Eigen::MatrixXd &x_1,
               const Eigen::MatrixXd &x_2,
               const double &learning_rate,
               const int &epochs);

    /**
     * @brief Train the model with a bias vector
     * @param x_1 Samples from the first class
     * @param x_2 Samples from the second class
     * @param b The bias vector
     */
    void Train(const Eigen::MatrixXd &x_1,
               const Eigen::MatrixXd &x_2,
               const Eigen::VectorXd &b);

    /**
     * @brief Predict the class of a sample
     * @param x The sample to predict
     * @return The predicted class
     */
    double Predict(Eigen::VectorXd x);
};

#endif  // LINEARCLASSIFIER_H