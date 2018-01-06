/**
 * @file LossTests.cpp
 *
 * @breif Tests for our loss functions
 *
 * @date 1/06/17
 * @author Ben Caine
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE LossTests

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "loss/CrossEntropy.h"
#include "loss/HuberLoss.h"
#include "loss/MeanSquaredError.h"

BOOST_AUTO_TEST_CASE(test_cross_entropy_loss) {
    // TODO: Really just checking compilation right now. Need to use TF/Pytorch/Numpy to generate test cases
    nn::CrossEntropyLoss<float, 2> lossFunction;

    int batchSize = 4;
    int numClasses = 3;
    Eigen::Tensor<float, 2> predictions(batchSize, numClasses);
    predictions.setValues({
                          {0.1, 0.7, 0.2},
                          {0.9, 0.0, 0.1},
                          {0.0, 0.0, 1.0},
                          {0.3, 0.4, 0.3}
                          });

    Eigen::Tensor<float, 2> labels(batchSize, numClasses);
    labels.setValues({
                             {0, 1, 0},
                             {1, 0, 0},
                             {1, 0, 0},
                             {0, 1, 0}
                     });

    auto loss = lossFunction.loss(predictions, labels);
    auto accuracy = lossFunction.accuracy(predictions, labels);
    auto backwardsResult = lossFunction.backward(predictions, labels);
    std::cout << "Loss : " << loss << " accuracy: " << accuracy << std::endl;
    BOOST_REQUIRE_MESSAGE(accuracy == 0.75, "Accuracy was not correct");
}

BOOST_AUTO_TEST_CASE(test_mse_loss) {
    nn::MeanSquaredError<float, 2> lossFunction;

    const int batchSize = 4;

    Eigen::Tensor<float, 2> predictions(batchSize, 1);
    predictions.setValues({{2}, {3}, {4}, {5}});

    Eigen::Tensor<float, 2> labels(batchSize, 1);
    labels.setValues({{2}, {1}, {3}, {0}});

    // Expected squared error of each should be:
    // 0^2 + 2^2 + 1^2 + 5^2 = 0 + 4 + 1 + 25 = 30

    auto loss = lossFunction.loss(predictions, labels);
    BOOST_REQUIRE_MESSAGE(loss == 30.0, "Loss not what was expected");

    auto backwardsResult = lossFunction.backward(predictions, labels);
    std::array<float, batchSize> expectedBackwardsResults = {0, 2, 1, 5};

    for (int ii = 0; ii < batchSize; ++ii) {
        BOOST_REQUIRE_CLOSE(backwardsResult(ii, 0), expectedBackwardsResults[ii], 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(test_huber_loss) {
    float threshold = 1.5;
    nn::HuberLoss<float, 2> lossFunction(threshold);

    const int batchSize = 4;
    Eigen::Tensor<float, 2> predictions(batchSize, 1);
    predictions.setValues({{2}, {3}, {4}, {5}});

    Eigen::Tensor<float, 2> labels(batchSize, 1);
    labels.setValues({{2}, {1}, {3}, {0}});

    // Expected absolute error:
    // [0, 2, 1, 5]
    // If our thresh is 1.5, then two terms are squared loss and two are absolute
    // So, we expect:
    // 0.5 * 0^2 + (1.5 * 2 - 0.5 * 1.5^2) + 0.5 * 1^2 + (1.5 * 5 - 0.5 * 1.5^2)
    // Which is:
    // [0, 1.875, 0.5, 6.375] = 8.75

    auto loss = lossFunction.loss(predictions, labels);
    BOOST_REQUIRE_CLOSE(loss, 8.75, 1e-3);

    auto backwardsResult = lossFunction.backward(predictions, labels);


    std::array<float, batchSize> expectedBackwardsResults = {0, threshold, 1, threshold};

    for (int ii = 0; ii < batchSize; ++ii) {
        BOOST_REQUIRE_CLOSE(backwardsResult(ii, 0), expectedBackwardsResults[ii], 1e-3);
    }
}