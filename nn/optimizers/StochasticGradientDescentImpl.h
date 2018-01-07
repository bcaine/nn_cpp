/**
 * @file StochasticGradientDescent.h
 *
 * @breif Stochastic Gradient Descent
 *
 * @date 1/06/17
 * @author Ben Caine
 */

#ifndef NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H
#define NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H

#include "OptimizerImpl.h"

namespace nn {
    namespace internal {
        template<typename Dtype, int Dims>
        class StochasticGradientDescentImpl : public OptimizerImpl<Dtype, Dims> {
        public:

            // TODO: Add momentum
            /**
             * @brief Initialize our SGD Solver
             * @param learningRate [in]: The learning rate of SGD
             */
            explicit StochasticGradientDescentImpl(Dtype learningRate):
                    m_learningRate(learningRate) {}

            /**
             * @brief Get the update to apply to the weights
             * @param gradWeights [in]: Weights to update
             * @return The factor to update the weights by
             */
            Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &gradWeights) {
                return gradWeights * gradWeights.constant(m_learningRate);
            };

        private:
            Dtype m_learningRate; ///< Our current learning rate
        };
    }
}

#endif //NN_CPP_STOCHASTICGRADIENTDESCENT_IMPL_H
