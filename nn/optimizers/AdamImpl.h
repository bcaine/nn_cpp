/**
 * @file AdamImpl.h
 *
 * @breif Adam Optimizer
 *
 * @date 1/06/17
 * @author Ben Caine
 */

#ifndef NN_CPP_ADAM_IMPL_H
#define NN_CPP_ADAM_IMPL_H

#include "OptimizerImpl.h"

namespace nn {
    namespace internal {
        template<typename Dtype, int Dims>
        class AdamImpl : public OptimizerImpl<Dtype, Dims> {
        public:

            /**
             * @brief Initialize our Adam Solver
             * @param learningRate [in]: Base learning rate
             * @param beta1 [in]: The first moment decay factor (default = 0.9)
             * @param beta2 [in]: The second moment decay factor (default = 0.999)
             * @param epsilon [in]: A stabilizing factor for division (default = 1e-8)
             */
            explicit AdamImpl(Dtype learningRate, Dtype beta1, Dtype beta2, Dtype epsilon):
                    m_learningRate(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon),
                    m_isInitialized(false), m_currentTimestep(1)
            {}

            /**
             * @brief Get the update to apply to the weights
             * @param gradWeights [in]: Weights to update
             * @return The factor to update the weights by
             */
            Eigen::Tensor<Dtype, Dims> weightUpdate(const Eigen::Tensor<Dtype, Dims> &gradWeights) {
                if (!m_isInitialized) {
                    m_firstMoment = Eigen::Tensor<Dtype, Dims>(gradWeights.dimensions());
                    m_firstMoment.setZero();

                    m_secondMoment = Eigen::Tensor<Dtype, Dims>(gradWeights.dimensions());
                    m_secondMoment.setZero();
                    m_isInitialized = true;
                }

                // m_t = B_1 * m_(t-1) + (1 - B_1) * g_t
                m_firstMoment = m_firstMoment.constant(m_beta1) * m_firstMoment +
                        gradWeights.constant(1 - m_beta1) * gradWeights;

                // v_t = B_2 * v_(t-1) + (1 - B_2) * g_t^2
                m_secondMoment = m_secondMoment.constant(m_beta2) * m_secondMoment +
                        gradWeights.constant(1 - m_beta2) * gradWeights.square();
//
//                std::cout << "First moment: " << m_firstMoment << std::endl;
//                std::cout << "Second moment: " << m_secondMoment << std::endl;
//                std::cout << std::endl << std::endl << std::endl;

                auto biasCorrectedFirstMoment = m_firstMoment / m_firstMoment.constant(1 - pow(m_beta1, m_currentTimestep));
                auto biasCorrectedSecondMoment = m_secondMoment / m_secondMoment.constant(1 - pow(m_beta2, m_currentTimestep));
//
//                std::cout << "Bias corrected first: " << biasCorrectedFirstMoment << std::endl;
//                std::cout << "Bias corrected second: " << biasCorrectedSecondMoment << std::endl;
//                std::cout << std::endl << std::endl << std::endl;


                m_currentTimestep ++;
                // Return firstMoment  * (learning_rate) / (sqrt(secondMoment) + epsilon)
                return biasCorrectedFirstMoment * (
                              (gradWeights.constant(m_learningRate) /
                               (biasCorrectedSecondMoment.sqrt() + gradWeights.constant(m_epsilon))
                ));
            };

        private:
            Dtype m_learningRate; ///< The learning rate of our optimizer
            Dtype m_beta1;        ///< Our B1 parameter (first moment decay)
            Dtype m_beta2;        ///< Our B2 parameter (second moment decay)
            Dtype m_epsilon;      ///< Stability factor

            bool m_isInitialized;      ///< On our first iteration, set the first and second order gradients to zero
            size_t m_currentTimestep;  ///< Our current timestep (iteration)

            // Our exponentially decaying average of past gradients
            Eigen::Tensor<Dtype, Dims> m_firstMoment;  ///< Our m_t term that represents the first order gradient decay
            Eigen::Tensor<Dtype, Dims> m_secondMoment; ///< Our v_t term that represents the second order gradient decay
        };
    }
}

#endif //NN_CPP_ADAM_IMPL_H
