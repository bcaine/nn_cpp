/**
 * @file Softmax.h
 *
 * @breif A Softmax layer
 *
 * @date 12/17/17
 * @author Ben Caine
 */
#ifndef NN_CPP_SOFTMAX_H
#define NN_CPP_SOFTMAX_H

#include "layers/Layer.h"

namespace nn {
    template <typename Dtype = float, int Dims = 2>
    class Softmax : public Layer<Dtype, Dims> {
    public:

        /**
         * @brief initialize Softmax
         */
        Softmax() = default;

        /**
         * @brief Forward through the layer (compute the output)
         * @param input [in]: The input tensor to apply softmax to
         * @return
         */
        Eigen::Tensor<Dtype, Dims> forward(const Eigen::Tensor<Dtype, Dims> &input);

        /**
         * @brief Compute the gradient (backwards pass) of the layer
         * @param input [in]: The input tensor to the backwards pass (from the next layer)
         * @return The output of the backwards pass (sent ot the previous layer)
         */
        Eigen::Tensor<Dtype, Dims> backward(const Eigen::Tensor<Dtype, Dims> &input);

        void printOutputShape() {}
    };

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Softmax<Dtype, Dims>::forward(const Eigen::Tensor<Dtype, Dims> &input) {
        int batchSize = input.dimensions()[0];
        int classDims = input.dimensions()[1];
        auto shiftedInput = input - input.maximum(Eigen::array<int, 1>{1})
                                    .eval().reshape(Eigen::array<int, 2>{batchSize, 1})
                                    .broadcast(Eigen::array<int, 2>{1, classDims});

        auto exponentiated = shiftedInput.exp();
        return exponentiated * exponentiated.sum(Eigen::array<int, 1>{1})
                               .inverse().eval()
                               .reshape(Eigen::array<int, 2>({batchSize, 1}))
                               .broadcast(Eigen::array<int, 2>({1, classDims}));
    }

    template <typename Dtype, int Dims>
    Eigen::Tensor<Dtype, Dims> Softmax<Dtype, Dims>::backward(const Eigen::Tensor<Dtype, Dims> &input) {

    }
}

#endif //NN_CPP_SOFTMAX_H
