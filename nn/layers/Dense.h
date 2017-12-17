/**
 * @file Dense.h
 *
 * @breif A Fully connected (Dense) layer
 *
 * @date 12/17/17
 * @author Ben Caine
 */

#ifndef NN_CPP_DENSE_H
#define NN_CPP_DENSE_H

#include "Layer.h"

namespace nn {
    template <typename Dtype = float, int Dims = 2>
    class Dense : public Layer<Dtype, Dims> {
    public:
        /**
         * @brief Create a Dense layer
         * @param outputShape [in]: The output dimensionality
         * @param inputShape [in]: The input dimensionality
         * @param useBias [in]: Whether to use a bias term
         */
        explicit Dense(const Eigen::array<Eigen::Index, Dims> &outputShape,
                       const Eigen::array<Eigen::Index, Dims> &inputShape,
                       bool useBias);

        /**
         * @brief Forward through the layer (compute the output)
         * @param input [in]: The input to the layer (either data or previous layer output)
         * @return The output of this layer
         */
        Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input);

        /**
         * @brief Compute the gradient (backward pass) of the layer
         * @param input [in]: The input to the backwards pass. (from next layer)
         * @return The output of the backwards pass (sent to previous layer)
         */
        Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &input);

        /**
         * @brief Get the input shape
         * @return The input shape
         */
        Eigen::array<Eigen::Index, Dims> getInputShape() {
            return m_inputShape;
        };

        Eigen::array<Eigen::Index, Dims> getOutputShape() {
            return m_outputShape;
        };

        void printInputShape() {
            std::cout << "[";
            for (const auto &dim : m_inputShape) {
                std::cout << dim << " ";
            }
            std::cout << "]";
        }

        void printOutputShape() {
            std::cout << "[";
            for (const auto &dim : m_outputShape) {
                std::cout << dim << " ";
            }
            std::cout << "]" << std::endl;
        }

    private:
        Eigen::array<Eigen::Index, Dims> m_outputShape; ///< The output shape of this layer
        Eigen::array<Eigen::Index, Dims> m_inputShape;  ///< The input shape if this layer
        Eigen::Tensor<Dtype, Dims> m_weights; ///< Our weights of the layer
        Eigen::Tensor<Dtype, Dims> m_bias;    ///< The bias weights if specified
        bool m_useBias;                       ///< Whether we use the bias
    };

    template <typename Dtype, int Dims>
    Dense<Dtype, Dims>::Dense(const Eigen::array<Eigen::Index, Dims> &outputShape,
                              const Eigen::array<Eigen::Index, Dims> &inputShape,
                              bool useBias):
            m_outputShape(outputShape),
            m_inputShape(inputShape),
            m_useBias(useBias)
    {
        m_weights = Eigen::Tensor<Dtype, Dims>(outputShape);
        m_weights.setRandom();

        if (useBias) {
            m_bias = Eigen::Tensor<Dtype, Dims>(outputShape);
            m_bias.setRandom();
        }
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Dense<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input) {
        // TODO: Add dimension checks
        Eigen::array<Eigen::IndexPair<int>, 1> productDims = { Eigen::IndexPair<int>(1, 0) };
        auto result = input.contract(m_weights, productDims);
        if (m_useBias) {
            return result + m_bias;
        }
        return result;
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Dense<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &input) {
        return input;
    }

}
#endif //NN_CPP_DENSE_H
