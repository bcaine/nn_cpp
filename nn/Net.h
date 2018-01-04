/**
 * @file Net.h
 *
 * @breif A basic Net class that provides an interface to a neural network
 *
 * @date 12/17/17
 * @author Ben Caine
 */

#ifndef NN_CPP_NET_H
#define NN_CPP_NET_H

#include "layers/Layers.h"
#include <vector>
#include <memory>


namespace nn {

    /**
     * @brief A neural network class
     */
    template <typename Dtype = float>
    class Net {
    public:
        /**
         * @brief Init a neural network wrapper
         */
        Net() = default;

        template <int inputDim, int outputDim>
        Eigen::Tensor<Dtype, outputDim> forward(Eigen::Tensor<Dtype, inputDim> input) {
            if (m_layers.empty()) {
                std::cerr << "No layers specified" << std::endl;
                return {};
            }

            // TODO: How to ensure each forward call returns a lazily evaluated expression instead of a Tensor
            // That way we can use this to autogenerate the evaluation chain for efficiency.
            // Right now it seems to evaluate each layer individually.
            auto currentInput = input;
            for (const auto &layer : m_layers) {
                currentInput = layer->forward(currentInput);
            }
            return currentInput;
        }

        template <int labelDims>
        void backward(Eigen::Tensor<Dtype, labelDims> input) {
            if (m_layers.empty()) {
                std::cerr << "No layers specified" << std::endl;
                return;
            }

            auto accumulatedGrad = input;
            for (auto rit = m_layers.rbegin(); rit != m_layers.rend(); ++rit) {
                accumulatedGrad = (*rit)->backward(accumulatedGrad);
            }
        }

        /**
         * @brief Update weights for each layer
         * @param learningRate [in]: Learning rate with amount to update
         */
        void updateWeights(float learningRate) {
            for (auto &layer : m_layers) {
                layer->updateWeights(learningRate);
            }
        }

        /**
         * @brief Add a layer to the neural network
         * @param layer [in]: A layer to add
         * @return A reference to *this for method chaining
         */
        template <int Dims>
        Net<Dtype>& add(std::unique_ptr<Layer<Dtype, Dims>> layer) {
            m_layers.push_back(layer);
            return *this;
        }

        /**
         * Add a dense layer
         * @param denseLayer [in]: The dense layer to add
         * @return A reference to *this for method chaining
         */
        template <int Dims>
        Net<Dtype>& add(Dense<Dtype, Dims> *denseLayer) {
            // Do shape checks here
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(denseLayer));
            return *this;
        }

        /**
         * Add a relu layer
         * @param reluLayer [in]: The relu layer to add
         * @return A reference to *this for method chaining
         */
        template <int Dims>
        Net<Dtype>& add(Relu<Dtype, Dims> *reluLayer) {
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(reluLayer));
            return *this;
        }

        /**
         * Add a softmax layer
         * @param softmaxLayer [in]: The softmax layer to add
         * @return A reference to *this for method chaining
         */
        template <int Dims>
        Net<Dtype>& add(Softmax<Dtype, Dims> *softmaxLayer) {
            m_layers.push_back(std::unique_ptr<Layer<Dtype, Dims>>(softmaxLayer));
            return *this;
        }

        void printShapes();

    private:
        std::vector<std::unique_ptr<Layer<Dtype>>> m_layers; ///< A vector of all our layers
    };

    template <typename Dtype>
    void Net<Dtype>::printShapes() {
        int layerNum = 0;
        for (const auto &layer : m_layers) {
            std::cout << "Layer " << layerNum << " Output Shape: ";
            layer->printOutputShape();
            std::cout << std::endl;
            layerNum ++;
        }
    }
}

#endif //NN_CPP_NET_H
