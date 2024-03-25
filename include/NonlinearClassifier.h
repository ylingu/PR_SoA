#include <memory>
#include <Eigen/Dense>

template <typename T>
struct TreeNode {
    T data;
    int split;
    int label;
    std::unique_ptr<TreeNode<T>> left;
    std::unique_ptr<TreeNode<T>> right;
    TreeNode(T data, int split, int label)
        : data(data), split(split), label(label), left(nullptr), right(nullptr)
    {
    }
};

class kNNClassifier
{
public:
    kNNClassifier(int k) : k(k) {}
    auto fit(const std::vector<std::pair<Eigen::VectorXd, int>> &data) -> void;
    auto predict(const std::vector<Eigen::VectorXd> &data) const -> std::vector<int>;

private:
    int k;
    std::unique_ptr<TreeNode<Eigen::VectorXd>> root;
};