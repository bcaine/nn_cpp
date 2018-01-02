/**
 * @file Relu.h
 *
 * @breif A Relu layer
 *
 * @date 12/17/17
 * @author Ben Caine
 */

#ifndef NN_CPP_RELU_H
#define NN_CPP_RELU_H

#include "layers/Layer.h"

namespace nn {
    template <typename Dtype = float, int Dims = 2>
    class Relu : public Layer<Dtype, Dims> {
    public:

        /**
         * @brief initialize Relu
         */
        Relu() = default;

        /**
         * @brief Return the name of the layer
         * @return The layer name
         */
        const std::string& getName() {
            const static std::string name = "Relu";
            return name;
        }

        /**
         * @brief Forward through the layer (compute the output)
         * @param input [in]: The input tensor to apply Relu to
         * @return max(0, input)
         */
        Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input);

        /**
         * @brief Compute the gradient (backward pass) of the layer
         * @param input [in]: The input to the backwards pass (from the next layer)
         * @return The output of the backwards pass (sent to the previous layer)
         */
        Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &input);

        /**
         * @brief Void function in relu
         */
        void updateWeights(float learningRate) {}

        void printOutputShape() {}
    private:
        Eigen::Tensor<Dtype, Dims> m_output; ///< The output of the forward pass
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Relu<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input) {
        return input.cwiseMax(static_cast<Dtype>(0));
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Relu<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &input) {
        // TODO: Hack to keep the weights from dying
        // For now we check if >= -0.01 so that slightly negative gradients work...
        // This is sort of a mash up between Relu and Leaky Relu, but seems to be working pretty well
        // on toy datasets.

        auto ifAlmostPositive = input >= static_cast<Dtype>(-0.01);
        return ifAlmostPositive.select(input, input.constant(0.0));
    }
}

#endif //NN_CPP_RELU_H
