/**
 * @file MeanSquaredError.h
 *
 * @breif Mean Squared Error loss
 *
 * @date 12/26/17
 * @author Ben Caine
 */
#ifndef NN_CPP_MEANSQUAREDERROR_H
#define NN_CPP_MEANSQUAREDERROR_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace nn {
    template<typename Dtype, int Dims>
    class MeanSquaredError {
    public:

        /**
         * @brief Initialize a Mean Squared Error loss function
         */
        MeanSquaredError() = default;

        /**
         * @brief Compute the MSE loss
         * @param predictions [in]: Predictions from the network
         * @param labels [in]: Labels to compute loss with
         * @return The loss as a scalar
         */
        Dtype loss(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

        /**
         * @brief Compute the gradient of Mean Squared Error given this data
         * @param predictions [in]: Predictions from the network
         * @param labels [in]: Labels from dataset
         * @return The gradient of the loss layer
         */
        Eigen::Tensor<Dtype, Dims>
        backward(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);
    };

    template<typename Dtype, int Dims>
    Dtype MeanSquaredError<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &predictions,
                                              const Eigen::Tensor<Dtype, Dims> &labels) {
        assert(predictions.dimensions()[0] == labels.dimensions()[0] &&
               "MeanSquaredError::loss dimensions don't match");
        assert(predictions.dimensions()[1] == labels.dimensions()[1] &&
               "MeanSquaredError::loss dimensions don't match");

        int batchSize = predictions.dimensions()[0];
        int numClasses = predictions.dimensions()[1];

        Eigen::Tensor<Dtype, 0> squaredSum = (predictions - labels).square().sum();
        return squaredSum(0) / numClasses;
    }

    template<typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> MeanSquaredError<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &predictions,
                                                                       const Eigen::Tensor<Dtype, Dims> &labels) {
        return predictions - labels;
    }
}

#endif //NN_CPP_MEANSQUAREDERROR_H
