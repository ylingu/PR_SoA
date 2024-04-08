#include "nonlinear_classifier.h"

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"

auto KNNClassifier::Fit(
    const std::vector<std::pair<Eigen::VectorXd, int>> &train_data) -> void {
    // 寻找方差最大的维度
    double max_var = 0;
    int split = 0;
    for (int i = 0; i < train_data[0].first.size(); i++) {
        double mean = 0;
        for (int j = 0; j < train_data.size(); j++) {
            mean += train_data[j].first(i);
        }
        mean /= train_data.size();
        double var = 0;
        for (int j = 0; j < train_data.size(); j++) {
            var += (train_data[j].first(i) - mean) *
                   (train_data[j].first(i) - mean);
        }
        if (var > max_var) {
            max_var = var;
            split = i;
        }
    }
    // 递归构建kd树
    std::function<void(std::unique_ptr<TreeNode<Eigen::VectorXd>> &,
                       const std::vector<std::pair<Eigen::VectorXd, int>> &,
                       int)>
        build = [&](std::unique_ptr<TreeNode<Eigen::VectorXd>> &current,
                    const std::vector<std::pair<Eigen::VectorXd, int>> &data,
                    int split) {
            if (data.size() == 0) {
                return;
            }
            // 寻找中位数
            auto sorted_data = data;
            int mid = sorted_data.size() / 2;
            std::sort(sorted_data.begin(), sorted_data.end(),
                      [split](const std::pair<Eigen::VectorXd, int> &a,
                              const std::pair<Eigen::VectorXd, int> &b) {
                          return a.first(split) < b.first(split);
                      });
            current = std::make_unique<TreeNode<Eigen::VectorXd>>(
                sorted_data[mid].first, split, sorted_data[mid].second);
            std::vector<std::pair<Eigen::VectorXd, int>> left_data(
                sorted_data.begin(), sorted_data.begin() + mid);
            std::vector<std::pair<Eigen::VectorXd, int>> right_data(
                sorted_data.begin() + mid + 1, sorted_data.end());
            build(current->left, left_data, (split + 1) % data[0].first.size());
            build(current->right, right_data,
                  (split + 1) % data[0].first.size());
        };
    build(root_, train_data, split);
}

auto KNNClassifier::Predict(const std::vector<Eigen::VectorXd> &test_data) const
    -> std::vector<int> {
    std::vector<int> result;
    // 递归搜索kd树
    std::function<void(const std::unique_ptr<TreeNode<Eigen::VectorXd>> &,
                       const Eigen::VectorXd &,
                       std::vector<std::pair<double, int>> &, int)>
        search = [&](const std::unique_ptr<TreeNode<Eigen::VectorXd>> &current,
                     const Eigen::VectorXd &data,
                     std::vector<std::pair<double, int>> &heap, int k) {
            if (current == nullptr) {
                return;
            }
            auto dist = (current->data - data).norm();
            if (heap.size() < k) {
                heap.push_back({dist, current->label});
                std::push_heap(heap.begin(), heap.end(),
                               [](const std::pair<double, int> &a,
                                  const std::pair<double, int> &b) {
                                   return a.first < b.first;
                               });
            } else {
                if (dist < heap.front().first) {
                    std::pop_heap(heap.begin(), heap.end(),
                                  [](const std::pair<double, int> &a,
                                     const std::pair<double, int> &b) {
                                      return a.first < b.first;
                                  });
                    heap.pop_back();
                    heap.push_back({dist, current->label});
                    std::push_heap(heap.begin(), heap.end(),
                                   [](const std::pair<double, int> &a,
                                      const std::pair<double, int> &b) {
                                       return a.first < b.first;
                                   });
                }
            }
            auto split_dist =
                std::abs(data(current->split) - current->data(current->split));
            if (heap.size() < k || split_dist < heap.front().first) {
                if (data(current->split) < current->data(current->split)) {
                    search(current->left, data, heap, k);
                    if (heap.size() < k || split_dist < heap.front().first) {
                        search(current->right, data, heap, k);
                    }
                } else {
                    search(current->right, data, heap, k);
                    if (heap.size() < k || split_dist < heap.front().first) {
                        search(current->left, data, heap, k);
                    }
                }
            }
        };
    for (int i = 0; i < test_data.size(); i++) {
        std::vector<std::pair<double, int>> heap;
        search(root_, test_data[i], heap, k_);
        std::map<int, int> count;
        for (int j = 0; j < heap.size(); j++) {
            count[heap[j].second]++;
        }
        int max_count = 0;
        int max_label = 0;
        for (auto &pair : count) {
            if (pair.second > max_count) {
                max_count = pair.second;
                max_label = pair.first;
            }
        }
        result.push_back(max_label);
    }
    return result;
}

int SVMClassifier::kEpochs = 1000;
double SVMClassifier::kEpsilon = 0.0001;

auto SVMClassifier::Train(const std::vector<cv::Mat> &train_data,
                          const std::vector<int> &train_label,
                          const std::vector<cv::Mat> &validate_data,
                          const std::vector<int> &validate_label,
                          const std::vector<double> &c_range,
                          const std::vector<double> &gamma_range,
                          const int &class_num) -> void {
    auto train_features64 = preprocessor_.Preprocessing(train_data),
         validate_features64 = preprocessor_.Preprocessing(validate_data);
    Eigen::MatrixXf train_features = train_features64.cast<float>(),
                    validate_features = validate_features64.cast<float>();
    cv::Mat train_data_mat, validate_data_mat;
    cv::eigen2cv(train_features, train_data_mat);
    cv::eigen2cv(validate_features, validate_data_mat);
    auto train_label_mat = cv::Mat(train_label, true);
    auto best_param = std::pair<double, double>(0, 0);
    auto best_accuracy = 0.0;
    for (auto c : c_range) {
        for (auto gamma : gamma_range) {
            auto temp = svm_;
            svm_ = cv::ml::SVM::create();
            svm_->setType(cv::ml::SVM::C_SVC);
            svm_->setKernel(cv::ml::SVM::RBF);
            svm_->setC(c);
            svm_->setGamma(gamma);
            svm_->setTermCriteria(cv::TermCriteria(
                cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kEpochs,
                kEpsilon));
            svm_->train(train_data_mat, cv::ml::ROW_SAMPLE, train_label_mat);
            auto result = Predict(validate_data_mat);
            auto accuracy = CalcAccuracy(result, validate_label);
            if (accuracy >= best_accuracy) {
                best_accuracy = accuracy;
                best_param = {c, gamma};
            } else {
                svm_ = temp;
            }
        }
    }
}

auto SVMClassifier::Predict(const cv::Mat &test_data_mat) const
    -> std::vector<int> {
    cv::Mat result;
    svm_->predict(test_data_mat, result);
    return std::vector<int>(result.begin<float>(), result.end<float>());
}

auto SVMClassifier::SaveModel(const std::string &filepath) -> void {
    svm_->save(filepath);
}

auto SVMClassifier::LoadModel(const std::string &filepath) -> void {
    svm_ = cv::ml::SVM::load(filepath);
}