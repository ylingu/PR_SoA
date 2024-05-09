#ifndef UNSUPERVISED_LEARNING_H
#define UNSUPERVISED_LEARNING_H

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "feature_extraction.h"

/**
 * @class Clustering
 * @brief An abstract class for clustering algorithms.
 */
class Clustering {
private:
    std::unique_ptr<FeatureExtraction>
        feature_extractor_;     ///< Feature extractor
    cv::Mat pca_eigen_vector_;  ///< PCA eigen vector

public:
    /**
     * @brief Construct a new Clustering object
     *
     * @param feature_extractor Unique pointer to the feature extractor
     */
    Clustering(std::unique_ptr<FeatureExtraction> feature_extractor)
        : feature_extractor_(std::move(feature_extractor)) {}

    /**
     * @brief Preprocess the data
     *
     * @param data Vector of data matrices
     * @param pca_dims Number of PCA dimensions (default is 0)
     * @return Eigen::MatrixXd Preprocessed data
     */
    auto Preprocess(const std::vector<cv::Mat> &data,
                    const int pca_dims = 0) -> Eigen::MatrixXd;

    /**
     * @brief Abstract method for clustering the data
     *
     * @param data Data to be clustered
     * @return Eigen::VectorXi Cluster labels
     */
    virtual auto Cluster(const Eigen::MatrixXd &data) -> Eigen::VectorXi = 0;

    /**
     * @brief Destroy the Clustering object (default)
     */
    virtual ~Clustering() = default;
};

/**
 * @class KmeansClustering
 * @brief Class for K-means clustering, derived from Clustering
 */
class KmeansClustering : public Clustering {
private:
    int k_;                           ///< Number of clusters
    cv::TermCriteria term_criteria_;  ///< Termination criteria
    int attempts_;                    ///< Number of attempts
    int flags_;                       ///< Flags

public:
    /**
     * @brief Construct a new Kmeans Clustering object
     *
     * @param feature_extractor Unique pointer to the feature extractor
     * @param k Number of clusters
     * @param term_criteria Termination criteria
     * @param attempts Number of attempts
     * @param flags Flags (default is cv::KMEANS_PP_CENTERS)
     */
    KmeansClustering(std::unique_ptr<FeatureExtraction> feature_extractor,
                     int k,
                     cv::TermCriteria term_criteria,
                     int attempts = 3,
                     int flags = cv::KMEANS_PP_CENTERS)
        : Clustering(std::move(feature_extractor)),
          k_(k),
          term_criteria_(term_criteria),
          attempts_(attempts),
          flags_(flags) {}

    /**
     * @brief Cluster the data using K-means
     *
     * @param data Data to be clustered
     * @return Eigen::VectorXi Cluster labels
     */
    auto Cluster(const Eigen::MatrixXd &data) -> Eigen::VectorXi override;
};

/**
 * @class DbscanClustering
 * @brief Class for DBSCAN clustering, derived from Clustering
 */
class DbscanClustering : public Clustering {
private:
    std::vector<Eigen::VectorXd> objects_ = {};  ///< Objects
    double eps_;                                 ///< Epsilon
    int min_samples_;                            ///< Minimum number of samples
    /**
     * @brief Get the neighbors of a point
     *
     * @param object Object
     * @return std::vector<Eigen::VectorXd> Neighbors
     */
    auto IsCoreObject(const Eigen::VectorXd &object) -> bool;

public:
    /**
     * @brief Construct a new Dbscan Clustering object
     *
     * @param feature_extractor Unique pointer to the feature extractor
     * @param eps Epsilon
     * @param min_samples Minimum number of samples
     */
    DbscanClustering(std::unique_ptr<FeatureExtraction> feature_extractor,
                     double eps,
                     int min_samples)
        : Clustering(std::move(feature_extractor)),
          eps_(eps),
          min_samples_(min_samples) {}

    /**
     * @brief Cluster the data using DBSCAN
     *
     * @param data Data to be clustered
     * @return Eigen::VectorXi Cluster labels
     */
    auto Cluster(const Eigen::MatrixXd &data) -> Eigen::VectorXi override;
};

#endif  // UNSUPERVISED_LEARNING_H