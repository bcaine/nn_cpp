/**
 * @file CrossEntropy.h
 *
 * @breif Cross Entropy loss
 *
 * @date 12/17/17
 * @author Ben Caine
 */

#ifndef NN_CPP_CROSSENTROPY_H
#define NN_CPP_CROSSENTROPY_H

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Dtype, int Dims>
class CrossEntropyLoss {
public:
    /**
     * @brief Create a cross entropy loss layer
     */
    CrossEntropyLoss() = default;

    /**
     * @brief Calculate the cross entropy loss
     * @param probabilities [in]: "Probabilities" as in 0-1 values output by a layer like Softmax
     * @param labels [in]: One hot encoded labels
     * @return The loss
     */
    Dtype loss(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);

    /**
     * @brief Calculate the accuracy of our labels
     * @param probabilities [in]: "Probabilities" as in 0-1 values output by a layer like Softmax
     * @param labels [in]: One hot encoded labels
     * @return The total accuracy (num_correct / total)
     */
    Dtype accuracy(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);

    /**
     * @brief Compute the gradient for Cross Entropy Loss
     * @param probabilities [in]: "Probabilities" as in 0-1 values output by a layer like Softmax
     * @param labels [in]: One hot encoded labels
     * @return The gradient of this loss layer
     */
    Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &probabilities, const Eigen::Tensor<Dtype, Dims> &labels);
};

template <typename Dtype, int Dims>
Dtype CrossEntropyLoss<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                          const Eigen::Tensor<Dtype, Dims> &labels) {
    Eigen::Tensor<Dtype, 0> summedLoss = (labels * probabilities.log()).sum();
    return -1.0 * summedLoss(0);
}

template <typename Dtype, int Dims>
Dtype CrossEntropyLoss<Dtype, Dims>::accuracy(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                              const Eigen::Tensor<Dtype, Dims> &labels) {
    assert(probabilities.dimensions()[0] == labels.dimensions()[0] && "CrossEntropy::accuracy dimensions did not match");
    assert(probabilities.dimensions()[1] == labels.dimensions()[1] && "CrossEntropy::accuracy dimensions did not match");

    int batchSize = labels.dimensions()[0];

    // Argmax across dimension = 1 (so we get a column vector)
    Eigen::Tensor<Eigen::DenseIndex, 0> count = (probabilities.argmax(1) * labels.argmax(1)).sum();
    return static_cast<Dtype>(count(0)) / batchSize;
}

template <typename Dtype, int Dims>
Eigen::Tensor<Dtype, Dims> CrossEntropyLoss<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &probabilities,
                                                                   const Eigen::Tensor<Dtype, Dims> &labels) {
    return probabilities - labels;
}
#endif //NN_CPP_CROSSENTROPY_H
