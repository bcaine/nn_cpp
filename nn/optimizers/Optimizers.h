/**
 * @file Optimizers.h
 *
 * @breif Optimizer Constructor
 *
 * @date 1/06/17
 * @author Ben Caine
 */

#ifndef NN_CPP_OPTIMIZERS_H
#define NN_CPP_OPTIMIZERS_H

#include "StochasticGradientDescentImpl.h"
#include <memory>

namespace nn {

    template <typename Dtype>
    class StochasticGradientDescent {
    public:
        explicit StochasticGradientDescent(Dtype learningRate):
            m_learningRate(learningRate) {}

        template <int Dims>
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> createOptimizer() const {
            return std::unique_ptr<OptimizerImpl<Dtype, Dims>>(new internal::StochasticGradientDescentImpl<Dtype, Dims>(m_learningRate));
        }

    private:
        Dtype m_learningRate; ///< The learning rate
    };
}

#endif //NN_CPP_OPTIMIZERS_H
