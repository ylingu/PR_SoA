#include "unsupervised_learning.h"
#include <Eigen/src/Core/Matrix.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/core/operations.hpp>

auto Clustering::Preprocess(const std::vector<cv::Mat> &data,
                            const int pca_dims) -> Eigen::MatrixXd {
    auto features = feature_extractor_->BatchExtract(data);
    if (pca_dims) {
        cv::Mat pca_data;
        cv::eigen2cv(features, pca_data);
        if (pca_eigen_vector_.empty()) {
            cv::PCA pca(pca_data, cv::Mat(), cv::PCA::DATA_AS_ROW, pca_dims);
            cv::transpose(pca.eigenvectors, pca_eigen_vector_);
        }
        cv::Mat pca_result = pca_data * pca_eigen_vector_;
        cv::cv2eigen(pca_result, features);
    }
    return features;
}

auto KmeansClustering::Cluster(const Eigen::MatrixXd &data) -> Eigen::VectorXi {
    cv::Mat data_mat;
    Eigen::MatrixXf data_f = data.cast<float>();
    cv::eigen2cv(data_f, data_mat);
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(
        data_mat, k_, labels, term_criteria_, attempts_, flags_, centers);
    Eigen::VectorXi cluster_labels(data.rows());
    for (int i = 0; i < data.rows(); ++i) {
        cluster_labels(i) = labels.at<int>(i);
    }
    return cluster_labels;
}

auto DbscanClustering::IsCoreObject(const Eigen::VectorXd &object) -> bool {
    int neighbor_num = 0;
    for (int i = 0; i < objects_.size(); ++i) {
        if ((object - objects_[i]).norm() < eps_) {
            neighbor_num++;
        }
    }
    return neighbor_num >= min_samples_;
}

auto DbscanClustering::Cluster(const Eigen::MatrixXd &data) -> Eigen::VectorXi {
    objects_.resize(data.rows());
    for (int i = 0; i < data.rows(); ++i) {
        Eigen::VectorXd object = data.row(i);
        objects_[i] = object;
    }
    std::vector<int> cluster_labels(data.rows(), -1);
    cv::partition(
        objects_,
        cluster_labels,
        [this](const Eigen::VectorXd &lhs, const Eigen::VectorXd &rhs) {
            if ((lhs - rhs).norm() < eps_) {
                return IsCoreObject(lhs) || IsCoreObject(rhs);
            }
            return false;
        });
    Eigen::VectorXi cluster_labels_eigen = Eigen::Map<Eigen::VectorXi>(
        cluster_labels.data(), cluster_labels.size());
    return cluster_labels_eigen;
}