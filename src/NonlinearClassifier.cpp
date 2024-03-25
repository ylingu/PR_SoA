#include "NonlinearClassifier.h"
#include <map>

auto kNNClassifier::fit(const std::vector<std::pair<Eigen::VectorXd, int>> &data) -> void
{
    //寻找方差最大的维度
    double maxVar = 0;
    int split = 0;
    for (int i = 0; i < data[0].first.size(); i++) {
        double mean = 0;
        for (int j = 0; j < data.size(); j++) {
            mean += data[j].first(i);
        }
        mean /= data.size();
        double var = 0;
        for (int j = 0; j < data.size(); j++) {
            var += (data[j].first(i) - mean) * (data[j].first(i) - mean);
        }
        if (var > maxVar) {
            maxVar = var;
            split = i;
        }
    }
    //递归构建kd树
    std::function<void(std::unique_ptr<TreeNode<Eigen::VectorXd>> &,
                       const std::vector<std::pair<Eigen::VectorXd, int>> &, int)>
        build = [&](std::unique_ptr<TreeNode<Eigen::VectorXd>> &current,
                    const std::vector<std::pair<Eigen::VectorXd, int>> &data, int split) {
            if (data.size() == 0) {
                return;
            }
            //寻找中位数
            auto sortedData = data;
            int mid = sortedData.size() / 2;
            std::sort(sortedData.begin(), sortedData.end(),
                      [split](const std::pair<Eigen::VectorXd, int> &a,
                              const std::pair<Eigen::VectorXd, int> &b) {
                          return a.first(split) < b.first(split);
                      });
            current = std::make_unique<TreeNode<Eigen::VectorXd>>(sortedData[mid].first, split,
                                                                  sortedData[mid].second);
            std::vector<std::pair<Eigen::VectorXd, int>> leftData(sortedData.begin(),
                                                                  sortedData.begin() + mid);
            std::vector<std::pair<Eigen::VectorXd, int>> rightData(sortedData.begin() + mid + 1,
                                                                   sortedData.end());
            build(current->left, leftData, (split + 1) % data[0].first.size());
            build(current->right, rightData, (split + 1) % data[0].first.size());
        };
    build(root, data, split);
}

auto kNNClassifier::predict(const std::vector<Eigen::VectorXd> &data) const -> std::vector<int>
{
    std::vector<int> result;
    //递归搜索kd树
    std::function<void(const std::unique_ptr<TreeNode<Eigen::VectorXd>> &, const Eigen::VectorXd &,
                       std::vector<std::pair<double, int>> &, int)>
        search = [&](const std::unique_ptr<TreeNode<Eigen::VectorXd>> &current,
                     const Eigen::VectorXd &data, std::vector<std::pair<double, int>> &heap,
                     int k) {
            if (current == nullptr) {
                return;
            }
            auto dist = (current->data - data).norm();
            if (heap.size() < k) {
                heap.push_back({dist, current->label});
                std::push_heap(heap.begin(), heap.end(),
                               [](const std::pair<double, int> &a,
                                  const std::pair<double, int> &b) { return a.first < b.first; });
            } else {
                if (dist < heap.front().first) {
                    std::pop_heap(
                        heap.begin(), heap.end(),
                        [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                            return a.first < b.first;
                        });
                    heap.pop_back();
                    heap.push_back({dist, current->label});
                    std::push_heap(
                        heap.begin(), heap.end(),
                        [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                            return a.first < b.first;
                        });
                }
            }
            auto splitDist = std::abs(data(current->split) - current->data(current->split));
            if (heap.size() < k || splitDist < heap.front().first) {
                if (data(current->split) < current->data(current->split)) {
                    search(current->left, data, heap, k);
                    if (heap.size() < k || splitDist < heap.front().first) {
                        search(current->right, data, heap, k);
                    }
                } else {
                    search(current->right, data, heap, k);
                    if (heap.size() < k || splitDist < heap.front().first) {
                        search(current->left, data, heap, k);
                    }
                }
            }
        };
    for (int i = 0; i < data.size(); i++) {
        std::vector<std::pair<double, int>> heap;
        search(root, data[i], heap, k);
        std::map<int, int> count;
        for (int j = 0; j < heap.size(); j++) {
            count[heap[j].second]++;
        }
        int maxCount = 0;
        int maxLabel = 0;
        for (auto &pair : count) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                maxLabel = pair.first;
            }
        }
        result.push_back(maxLabel);
    }
    return result;
}