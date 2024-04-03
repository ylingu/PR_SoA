#include "LinearClassifier.h"
#include "BayesClassifier.h"
#include "NonlinearClassifier.h"
#include <format>

using namespace Eigen;
using namespace std;

int main()
{
    // *定义两个类别的数据
    // MatrixXd X_1(2, 2);
    // X_1 << 0, 1, 0, 2;
    // MatrixXd X_2(2, 2);
    // X_2 << 1, 3, -1, 0;
    // Vector4d b;
    // b << 1, 1, 1, 1;
    // Vector2d x(5, 0);
    // *求最小平方误差准则的权重向量
    // LeastSquaresCriterion c;
    // LinearClassifier lc(&c);
    // lc.train(X_1, X_2, b);
    // *使用垂直平分分类器
    // VerticalBisectorClassifier vbc;
    // vbc.train(X_1, X_2);
    // cout << "vbc.predict(x) = " << vbc.predict(x) << endl;
    // *使用fisher准则
    // cout << "fisher(X_1, X_2) = " << fisher(X_1, X_2) << endl;
    // *使用kNN分类器
    kNNClassifier knn(1);
    vector<pair<VectorXd, int>> data;
    data.push_back(make_pair(Vector2d(2, 2), 0));
    data.push_back(make_pair(Vector2d(2, 3), 0));
    data.push_back(make_pair(Vector2d(1, 2), 0));
    data.push_back(make_pair(Vector2d(2, 1), 0));
    data.push_back(make_pair(Vector2d(-2, -2), 1));
    data.push_back(make_pair(Vector2d(-3, -2), 1));
    data.push_back(make_pair(Vector2d(-1, -2), 1));
    data.push_back(make_pair(Vector2d(-2, -3), 1));
    knn.fit(data);
    vector<VectorXd> test;
    test.push_back(Vector2d(-1, -1));
    test.push_back(Vector2d(3, 2));
    auto res = knn.predict(test);
    cout << format("result = {{{}, {}}}", res[0], res[1]) << endl;
    return 0;
}
