#include "LinearClassifier.h"

using namespace Eigen;
using namespace std;

int main()
{
    //定义两个类别的数据
    MatrixXd X_1(2, 2);
    X_1 << 0, 1, 0, 2;
    MatrixXd X_2(2, 2);
    X_2 << 1, 3, -1, 0;
    Vector4d b;
    b << 1, 1, 1, 1;
    Vector2d x(5, 0);
    // //求最小平方误差准则的权重向量
    // LeastSquaresCriterion c;
    // LinearClassifier lc(&c);
    // lc.train(X_1, X_2, b);
    // //使用垂直平分分类器
    // VerticalBisectorClassifier vbc;
    // vbc.train(X_1, X_2);
    // cout << "vbc.predict(x) = " << vbc.predict(x) << endl;
    //使用fisher准则
    cout << "fisher(X_1, X_2) = " << fisher(X_1, X_2) << endl;
    return 0;
}
