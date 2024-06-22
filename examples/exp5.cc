#include <opencv2/opencv.hpp>
#include <vector>

#include "csv.h"
#include "feature_extraction.h"
#include "layers.h"
#include "optimizer.h"

using Tensor = Eigen::Tensor<float, 2>;
class BaseMLP {
private:
    nn::Sequential<nn::Linear<Tensor>, nn::ReLU<Tensor>, nn::Linear<Tensor>>
        model_;

public:
    BaseMLP(int input_size, int hidden_size, int output_size)
        : model_(nn::Linear<Tensor>(input_size, hidden_size),
                 nn::ReLU<Tensor>(),
                 nn::Linear<Tensor>(hidden_size, output_size)) {}
    auto Preprocess(const std::vector<cv::Mat> &data)
        -> Eigen::Tensor<float, 2> {
        auto feature_extractor = GLCMFeatureExtraction();
        auto features = feature_extractor.BatchExtract(data);
        Eigen::MatrixXf features_f = features.cast<float>();
        Eigen::Tensor<float, 2> ret = Eigen::TensorMap<Eigen::Tensor<float, 2>>(
            features_f.data(), data.size(), features_f.cols());
        return ret;
    }
    auto TrainEpoch(const Eigen::Tensor<float, 2> &train_data,
                    const Eigen::Tensor<int, 1> &train_label,
                    nn::CrossEntropyLoss<Tensor> &loss,
                    optim::Adam &optimizer) -> float {
        auto output = model_.Forward(train_data);
        auto loss_val = loss.Forward(output, train_label);
        auto grad = loss.Backward();
        model_.Backward(grad);
        model_.Step(optimizer);
        return loss_val;
    }
    auto Predict(const Eigen::Tensor<float, 2> &test_data)
        -> Eigen::Tensor<Eigen::Index, 1> {
        return model_.Forward(test_data).argmax(1);
    }
};


int main() {
    std::vector<cv::Mat> train_imgs, test_imgs;
    std::vector<int> train_label, test_label;
    for (int i = 1; i < 21; ++i) {
        int j = 1;
        for (; j < 17; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            train_imgs.push_back(img);
            train_label.push_back(i - 1);
        }
        for (; j < 21; ++j) {
            auto img =
                cv::imread("../../../../data/imgs/s" + std::to_string(i) + "/" +
                               std::to_string(j) + ".bmp",
                           cv::IMREAD_GRAYSCALE);
            test_imgs.push_back(img);
            test_label.push_back(i - 1);
        }
    }
    BaseMLP mlp(16, 15, 20);
    auto train_data = mlp.Preprocess(train_imgs);
    auto test_data = mlp.Preprocess(test_imgs);
    auto train_label_tensor = Eigen::TensorMap<Eigen::Tensor<int, 1>>(
        train_label.data(), train_label.size());

    auto loss = nn::CrossEntropyLoss<Tensor>();
    auto optimizer = optim::Adam(0.01);
    std::vector<std::vector<std::string>> data{
        {"epoch", "train loss", "train accuracy", "test accuracy"}};
    CSV csv(data);
    for (int i = 0; i <= 3000; ++i) {
        auto loss_val =
            mlp.TrainEpoch(train_data, train_label_tensor, loss, optimizer);
        if (i % 100 == 0)
            csv.InsertRow(csv.GetRowCount(),
                          {std::to_string(i),
                           std::to_string(loss_val),
                           std::to_string(CalcAccuracy(mlp.Predict(train_data),
                                                       train_label)),
                           std::to_string(CalcAccuracy(mlp.Predict(test_data),
                                                       test_label))});
    }
    csv.Save("exp5.csv");
    return 0;
}