/**
 * @file Layer.h
 *
 * @breif A base class that provides an interface to a layer
 *
 * @date 12/17/17
 * @author Ben Caine
 */

#ifndef NN_CPP_LAYER_H
#define NN_CPP_LAYER_H

#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "optimizers/Optimizers.h"

namespace nn {
    template <typename Dtype = float, int Dims = 2>
    class Layer {
    public:
        /**
         * @brief Return the name of the layer
         */
        virtual const std::string& getName() = 0;

        /**
         * @brief Take an input tensor, perform an operation on it, and return a new tensor
         * @param input [in]: The input tensor (from the previous layer)
         * @return An output tensor, which is fed into the next layer
         */
        virtual Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input) = 0;

        /**
         * @brief Perform the backwards operation on the layer.
         * @param input [in]: The input tensor (from next layer)
         * @return The output tensor, which is fed into the previous layer
         */
        virtual Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &output) = 0;

        /**
         * @brief Update the weights after a backwards pass
         */
        virtual void step() = 0;

        /**
         * @brief Registers the optimizer with the layer
         * @param optimizer [in]: The optimizer to register
         */
        virtual void registerOptimizer(std::shared_ptr<StochasticGradientDescent<Dtype>> optimizer) = 0;
    };
}

#endif //NN_CPP_LAYER_H
