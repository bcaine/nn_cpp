/**
 * @file WeightInitializers.h
 *
 * @breif A collection of helper functions to initialize weights
 *
 * @date 12/17/17
 * @author Ben Caine
 */
#ifndef NN_CPP_WEIGHTINITIALIZERS_H
#define NN_CPP_WEIGHTINITIALIZERS_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <memory>

namespace nn {
    /**
     * @brief How to initialize the dense weights
     */
    enum class InitializationScheme {
        GlorotUniform,
        GlorotNormal
    };

    template <typename Dtype>
    class WeightDistribution {
    public:
        /**
         * @brief Create a weight distribution to draw from
         * @param scheme [in]: The scheme to initialize with
         * @param fanIn [in]: The fan in of the layer
         * @param fanOut [in]: The fan out of the layer
         */
        explicit WeightDistribution(InitializationScheme scheme, int fanIn, int fanOut):
                m_scheme(scheme),
                m_randomNumberGenerator(std::random_device()())
        {
            if (m_scheme == InitializationScheme::GlorotUniform) {
                Dtype limit = std::sqrt(6.0 / (fanIn + fanOut));
                m_uniformDist.reset(new std::uniform_real_distribution<Dtype>(-limit, limit));
            } else if (m_scheme == InitializationScheme::GlorotNormal) {
                Dtype std = std::sqrt(2.0 / (fanIn + fanOut));
                m_normalDist.reset(new std::normal_distribution<Dtype>(0, std));
            }
        }

        /**
         * @brief Get a value from the distribution
         * @return
         */
        Dtype get() {
            if (m_scheme == InitializationScheme::GlorotUniform) {
                return (*m_uniformDist)(m_randomNumberGenerator);
            } else if (m_scheme == InitializationScheme::GlorotNormal) {
                return (*m_normalDist)(m_randomNumberGenerator);
            } else {
                std::cerr << "Tried to draw from distribution that is uninitialized" << std::endl;
                exit(-1);
            }
        }

    private:
        InitializationScheme m_scheme;                                        ///< Our init scheme
        std::mt19937 m_randomNumberGenerator;                                 ///< Our random number generator
        std::unique_ptr<std::uniform_real_distribution<Dtype>> m_uniformDist; ///< Our uniform distribution
        std::unique_ptr<std::normal_distribution<Dtype>> m_normalDist;        ///< Our normal distribution
    };


    /**
     * @brief Initialize a tensor of dimension (input x output) with a specified scheme
     * @tparam Dtype [in]: Datatype of the tensor (float/double)
     * @param inputDimensions [in]: The input dimensions of the layer
     * @param outputDimensions [in]: The output dimensions of the layer
     * @param scheme [in]: Initialization Scheme
     * @return A randomly initialized tensor
     *
     * @note This function only exists because I can't seem to get Tensor.setRandom<Generator> to work
     *       with their builtins. This is way, way less efficient, but is only called on creation of a new layer
     */
    template <typename Dtype>
    Eigen::Tensor<Dtype, 2> getRandomWeights(int inputDimensions, int outputDimensions,
                                             InitializationScheme scheme = InitializationScheme::GlorotUniform) {
        Eigen::Tensor<Dtype, 2> weights(inputDimensions, outputDimensions);
        weights.setZero();

        auto distribution = WeightDistribution<Dtype>(scheme, inputDimensions, outputDimensions);
        for (unsigned int ii = 0; ii < inputDimensions; ++ii) {
            for (unsigned int jj = 0; jj < outputDimensions; ++jj) {
                weights(ii, jj) = distribution.get();
            }
        }
        return weights;
    };
}

#endif //NN_CPP_WEIGHTINITIALIZERS_H
