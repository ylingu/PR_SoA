#include "LinearClassifier.h"
#include "BayesClassifier.h"
#include "NonlinearClassifier.h"

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
    kNNClassifier knn(3);
    vector<pair<VectorXd, int>> data;
    data.push_back(make_pair(Vector2d(1.24, -2.86), 0));
    data.push_back(make_pair(Vector2d(-6.88, -5.40), 0));
    data.push_back(make_pair(Vector2d(-2.96, -0.50), 0));
    data.push_back(make_pair(Vector2d(-4.60, -10.55), 0));
    data.push_back(make_pair(Vector2d(-4.96, 12.61), 0));
    data.push_back(make_pair(Vector2d(1.75, 12.26), 0));
    data.push_back(make_pair(Vector2d(6.27, 5.50), 1));
    data.push_back(make_pair(Vector2d(17.05, -12.79), 1));
    data.push_back(make_pair(Vector2d(7.75, -22.68), 1));
    data.push_back(make_pair(Vector2d(10.80, -5.03), 1));
    data.push_back(make_pair(Vector2d(15.31, -13.16), 1));
    data.push_back(make_pair(Vector2d(7.83, 15.70), 1));
    data.push_back(make_pair(Vector2d(14.63, -0.35), 1));
    knn.fit(data);
    vector<VectorXd> test;
    test.push_back(VectorXd(2));
    test[0] << -1, -5;
    auto res = knn.predict(test);
    cout << "knn.predict(test) = " << res[0] << '\n';
    return 0;
}
