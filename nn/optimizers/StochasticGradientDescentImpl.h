/**
 * @file StochasticGradientDescent.h
 *
 * @breif Stochastic Gradient Descent
 *
 * @date 1/06/17
 * @author Ben Caine
 */

#ifndef NNCPP_STOCHASTICGRADIENTDESCENT_IMPL_H
#define NNCPP_STOCHASTICGRADIENTDESCENT_IMPL_H

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

            Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &weights) {
                return weights * weights.constant(m_learningRate);
            };

        private:
            Dtype m_learningRate; ///< Our current learning rate
        };
    }
}

#endif //NNCPP_STOCHASTICGRADIENTDESCENT_IMPL_H
