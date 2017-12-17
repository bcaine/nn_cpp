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

#include "layers/Layer.h"
#include "layers/Dense.h"
#include <vector>

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

            auto currentInput = input;
            for (const auto &layer : m_layers) {
                currentInput = layer->forward(currentInput);
            }
            return currentInput;
        }

        /**
         * @brief Add a layer to the neural network
         * @param layer [in]: A layer to add
         * @return A reference to *this for method chaining
         */
        Net<Dtype>& add(std::unique_ptr<Layer<Dtype>> layer);

        /**
         * Add a dense layer
         * @param denseLayer [in]: The dense layer to add
         * @return A reference to *this for method chaining
         */
        template <int Dims>
        Net<Dtype>& add(Dense<Dtype, Dims> *denseLayer) {
            m_layers.push_back(std::unique_ptr<Layer<Dtype>>(denseLayer));
            return *this;
        }

        void printShapes();


    private:
        std::vector<std::unique_ptr<Layer<Dtype>>> m_layers; ///< A vector of all our layers
    };

    template <typename Dtype>
    Net<Dtype>& Net<Dtype>::add(std::unique_ptr<Layer<Dtype>> layer) {
        // TODO: Do check about dimensions?
        m_layers.push_back(layer);
        return *this;
    }

    template <typename Dtype>
    void Net<Dtype>::printShapes() {
        int layerNum = 0;
        for (const auto &layer : m_layers) {
            std::cout << "Layer " << layerNum << std::endl;
            layer->printInputShape();
            std::cout << "    ->    ";
            layer->printOutputShape();
            layerNum ++;
        }
    }
}

#endif //NN_CPP_NET_H
