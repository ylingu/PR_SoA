#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class FeatureExtraction
 * @brief Abstract class for extracting features from images.
 *
 * Provides a common interface for feature extraction techniques such as GLCM,
 * Gabor, HOG, and LBP.
 */
class FeatureExtraction {
protected:
    static int kLevel;  // Level
public:
    /**
     * @brief Compresses an image using specified parameters.
     *
     * This static method can be used to compress an image before feature
     * extraction to reduce computation.
     *
     * @param img The image to be compressed.
     * @return The compressed cv::Mat image.
     */
    static auto Compress(const cv::Mat &img) -> cv::Mat;

    /**
     * @brief Pure virtual method to extract features from a single image.
     *
     * Must be implemented by derived classes to extract features according to
     * specific algorithms.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted features.
     */
    virtual auto Extract(const cv::Mat &img) -> Eigen::VectorXd = 0;

    /**
     * @brief Pure virtual method to extract features from a batch of images.
     *
     * Must be implemented by derived classes to extract features from multiple
     * images efficiently.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the features extracted
     * from one image.
     */
    virtual auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd = 0;

    /**
     * @brief Default destructor.
     */
    virtual ~FeatureExtraction() = default;
};

/**
 * @class GLCMFeatureExtraction
 * @brief GLCM (Gray Level Co-occurrence Matrix) feature extraction class.
 *
 * Inherits from FeatureExtraction and implements feature extraction using the
 * GLCM method. This class is designed to compute GLCM features such as
 * contrast, correlation, energy, and homogeneity from images. It supports
 * extracting features from both single images and batches of images.
 */
class GLCMFeatureExtraction : public FeatureExtraction {
private:
    int d_;  ///< Distance parameter for GLCM computation.
    std::vector<double>
        angles_;  ///< Angles to be considered in GLCM computation.
public:
    /**
     * @brief Constructs a GLCMFeatureExtraction object with specified angles
     * and distance.
     *
     * @param angles A vector of angles (in radians) to be used for GLCM
     * computation. Default angles are 0, 45, 90, and 135 degrees.
     * @param d The distance parameter for GLCM computation. Default is 1.
     */
    GLCMFeatureExtraction(const std::vector<double> &angles = {0.0,
                                                               CV_PI / 4,
                                                               CV_PI / 2,
                                                               CV_PI * 3 / 4},
                          int d = 1)
        : angles_(angles), d_(d) {}

    /**
     * @brief Extracts GLCM features from a single image.
     *
     * This method overrides the pure virtual method from the FeatureExtraction
     * class. It computes the GLCM for the given image and extracts features
     * such as contrast, correlation, energy, and homogeneity.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted GLCM features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts GLCM features from a batch of images.
     *
     * This method implements efficient extraction of GLCM features from
     * multiple images. It is particularly useful for processing large datasets
     * or real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the GLCM features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};

class GaborFeatureExtraction : public FeatureExtraction {
private:
    std::vector<cv::Mat> kernels_;  ///< Gabor kernels.
    std::vector<double> thetas_;    ///< Orientations for Gabor kernels.
    std::vector<double> lambdas_;   ///< Wavelengths for Gabor kernels.
    cv::Size kernel_size_;          ///< Size of the Gabor kernel.
    double gamma_;                  ///< Gamma value for Gabor kernel.
    double sigma_;                  ///< Sigma value for Gabor kernel.
    double psi_;                    ///< Psi value for Gabor kernel.
public:
    /**
     * @brief This class is used for extracting Gabor features from images.
     *
     * GaborFeatureExtraction utilizes Gabor filters for texture analysis in
     * images. It applies Gabor filters with specified orientations and
     * wavelengths to the input image and extracts features that are useful for
     * various image processing and computer vision tasks, such as texture
     * classification and object recognition.
     *
     * @param thetas A constant reference to a std::vector<double> containing
     * the orientations (in radians) for the Gabor filters. These orientations
     * determine the direction of the texture features to be extracted.
     * @param lambdas A constant reference to a std::vector<double> containing
     * the wavelengths (in pixels) of the sinusoidal wave component of the Gabor
     * filters. The wavelengths determine the scale of the texture features to
     * be extracted.
     * @param kernel_size Optional parameter specifying the size of the Gabor
     * kernel. Default is cv::Size(17, 17). The kernel size affects the locality
     * of the texture features being analyzed.
     * @param gamma Optional parameter specifying the spatial aspect ratio of
     * the Gabor kernel. Default value is 1. Gamma influences the ellipticity of
     * the support of the Gabor function.
     * @param sigma Optional parameter specifying the standard deviation of the
     * Gaussian envelope of the Gabor kernel. Default value is CV_PI. Sigma
     * controls the width of the Gaussian envelope, affecting the scale of
     * texture features that are emphasized.
     * @param psi Optional parameter specifying the phase offset of the
     * sinusoidal wave component of the Gabor kernel. Default value is 0. Psi
     * allows adjustment of the phase of the Gabor filter, which can influence
     * the response of the filter to certain textures.
     */
    GaborFeatureExtraction(const std::vector<double> &thetas,
                           const std::vector<double> &lambdas,
                           const cv::Size kernel_size = cv::Size(17, 17),
                           const double gamma = 1,
                           const double sigma = CV_PI,
                           const double psi = 0)
        : thetas_(thetas),
          lambdas_(lambdas),
          kernel_size_(kernel_size),
          gamma_(gamma),
          sigma_(sigma),
          psi_(psi) {}

    /**
     * @brief Generates Gabor kernels for a given set of orientations and
     * wavelengths.
     *
     * This function populates internal structures with Gabor kernels calculated
     * for the specified orientations and wavelengths. Each combination of theta
     * and lambda results in a unique Gabor kernel, which can be used for
     * texture analysis, feature extraction, and other image processing tasks.
     * The generated kernels are stored internally and can be accessed through
     * other class methods.
     */
    inline auto GetGaborKernel() -> void {
        for (auto theta : thetas_) {
            for (auto lambda : lambdas_) {
                cv::Mat kernel = cv::getGaborKernel(
                    kernel_size_, sigma_, theta, lambda, gamma_, 0, CV_32F);
                kernels_.push_back(kernel);
            }
        }
    }

    /**
     * @brief Extracts Gabor features from a single image.
     *
     * This method overrides the pure virtual method from the
     * FeatureExtraction class. It computes the Gabor features for the given
     * image.
     *
     * @param img The image from which features are to be extracted.
     * @return An Eigen::VectorXd containing the extracted Gabor features.
     */
    auto Extract(const cv::Mat &img) -> Eigen::VectorXd override;

    /**
     * @brief Extracts Gabor features from a batch of images.
     *
     * This method implements efficient extraction of Gabor features from
     * multiple images. It is particularly useful for processing large datasets
     * or real-time applications.
     *
     * @param data A vector of images from which features are to be extracted.
     * @return An Eigen::MatrixXd where each row contains the Gabor features
     * extracted from one image.
     */
    auto BatchExtract(const std::vector<cv::Mat> &data)
        -> Eigen::MatrixXd override;
};

#endif  // FEATURE_EXTRACTION_H