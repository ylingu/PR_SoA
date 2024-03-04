#include "LinearClassifier.h"

using namespace Eigen;
using namespace std;

int main()
{
    //定义两个类别的数据
    MatrixXd X_1(2, 2);
    X_1 << 0, 0, 0, 1;
    MatrixXd X_2(2, 2);
    X_2 << 2, 1, 0, 1;
    //训练分类器
    cout << fisher(X_1, X_2);
    return 0;
}
