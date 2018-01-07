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
#include "AdamImpl.h"
#include <memory>

namespace nn {
    /**
     * The current design is that you declare your optimizer in your main training area
     * and the Net class propagates this to all layers, which create their own Impls
     * with one Impl per weight. The design is geared towards more complex optimizers
     */


    /**
     * @brief Factory method of SGD
     *
     * @tparam Dtype : The floating point type of the optimizer
     */
    template <typename Dtype>
    class StochasticGradientDescent {
    public:
        /**
         * @brief Create a SGD factory w/ learning rate
         * @param learningRate [in]: Learning rate of SGD optimizer
         */
        explicit StochasticGradientDescent(Dtype learningRate):
            m_learningRate(learningRate) {}

        /**
         * @brief Create an optimizer Impl for our given type
         * @tparam Dims [in]: The dimensionality of the tensor the optimizer will update
         * @return An optimizer impl that can update weights and keep track of state
         */
        template <int Dims>
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> createOptimizer() const {
            return std::unique_ptr<OptimizerImpl<Dtype, Dims>>(new internal::StochasticGradientDescentImpl<Dtype, Dims>(m_learningRate));
        }

    private:
        Dtype m_learningRate; ///< The learning rate
    };

    template <typename Dtype>
    class Adam {
    public:
        /**
         * @brief Create an Adam optimizer
         * @param learningRate [in]: Base learning rate
         * @param beta1 [in]: The first moment decay factor (default = 0.9)
         * @param beta2 [in]: The second moment decay factor (default = 0.999)
         * @param epsilon [in]: A stabilizing factor for division (default = 1e-8)
         */
        explicit Adam(Dtype learningRate, Dtype beta1 = 0.9, Dtype beta2 = 0.999, Dtype epsilon = 1e-8):
                m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon)
        {}

        /**
         * Create an optimizer Impl for our given type
         * @tparam Dims [in]: The dimensionality of the tensor the optimizer will update
         * @return An optimizer impl that can update weights and keep track of state
         */
        template <int Dims>
        std::unique_ptr<OptimizerImpl<Dtype, Dims>> createOptimizer() const {
            return std::unique_ptr<OptimizerImpl<Dtype, Dims>>(new internal::AdamImpl<Dtype, Dims>(m_learningRate, m_beta1, m_beta2, m_epsilon));
        };

    private:
        Dtype m_learningRate; ///< The learning rate of our optimizer
        Dtype m_beta1;        ///< Our B1 parameter (first moment decay)
        Dtype m_beta2;        ///< Our B2 parameter (second moment decay)
        Dtype m_epsilon;      ///< Stability factor
    };


}

#endif //NN_CPP_OPTIMIZERS_H
