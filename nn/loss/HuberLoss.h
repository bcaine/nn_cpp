/**
 * @file HuberLoss.h
 *
 * @breif Huber Loss
 *
 * @date 1/06/17
 * @author Ben Caine
 */
#ifndef NN_CPP_HUBERLOSS_H
#define NN_CPP_HUBERLOSS_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace nn {
    template<typename Dtype, int Dims>
    class HuberLoss {
    public:

        /**
         * @brief Initialize a SmoothL1Loss loss function
         */
        explicit HuberLoss(Dtype threshold = 1.0): m_threshold(threshold) {}

        /**
         * @brief Compute the loss
         * @param predictions [in]: Predictions from the network
         * @param labels [in]: Labels to compute loss with
         * @return The loss as a scalar
         */
        Dtype loss(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

        /**
         * @brief Compute the gradient of the loss given this data
         * @param predictions [in]: Predictions from the network
         * @param labels [in]: Labels from dataset
         * @return The gradient of the loss layer
         */
        Eigen::Tensor<Dtype, Dims>
        backward(const Eigen::Tensor<Dtype, Dims> &predictions, const Eigen::Tensor<Dtype, Dims> &labels);

    private:
        Dtype m_threshold;                                ///< The threshold used to determine which part of the piecewise loss
        Eigen::Tensor<bool, Dims> m_cachedSwitchResults;  ///< Whether abs(y - y_hat) <= m_threshold
    };

    template<typename Dtype, int Dims>
    Dtype HuberLoss<Dtype, Dims>::loss(const Eigen::Tensor<Dtype, Dims> &predictions,
                                              const Eigen::Tensor<Dtype, Dims> &labels) {
        assert(predictions.dimensions()[0] == labels.dimensions()[0] &&
               "HuberLoss::loss dimensions don't match");
        assert(predictions.dimensions()[1] == labels.dimensions()[1] &&
               "HuberLoss::loss dimensions don't match");
        int batchSize = predictions.dimensions()[0];
        // Definition taken from: https://en.wikipedia.org/wiki/Huber_loss

        // Precalculate y_hat - y
        auto error = predictions - labels;
        auto absoluteError = error.abs();

        // Set up our switch statement and cache it
        m_cachedSwitchResults = absoluteError <= m_threshold;

        // Calculate both terms for the huber loss
        auto lessThanThreshold = error.constant(0.5) * error.square();
        auto moreThanThreshold = error.constant(m_threshold) * absoluteError - error.constant(0.5 * pow(m_threshold, 2));

        // If abs(y_hat - y) <= threshold
        auto perItemLoss = m_cachedSwitchResults.select(
                lessThanThreshold, // Then use 0.5 * (y_hat - y)^2
                moreThanThreshold); // Else use thresh * |y_hat - y| - (0.5 * threshold^2)

        Eigen::Tensor<Dtype, 0> sum = perItemLoss.sum();
        // Sum and divide by N
        return sum(0) / batchSize;
    }

    template<typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> HuberLoss<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &predictions,
                                                                       const Eigen::Tensor<Dtype, Dims> &labels) {

        auto error = predictions - labels;

        // Note: Grad of linear part of error is threshold * (error / abs(error)), which
        // simplifies to threshold * sign(error)
        auto errorPositiveOrZero = error >= static_cast<Dtype>(0);
        auto absoluteErrorGrad = errorPositiveOrZero.select(error.constant(m_threshold), error.constant(-m_threshold));
        return m_cachedSwitchResults.select(error, absoluteErrorGrad);
    }
}

#endif //NN_CPP_SMOOTHL1LOSS_H
