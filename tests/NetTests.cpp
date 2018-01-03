/**
 * @file NetTests.cpp
 *
 * @breif High level tests of the Net class
 *
 * @date 12/17/17
 * @author Ben Caine
 */


#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE NetTests

#include <boost/test/unit_test.hpp>
#include <chrono>
#include "Net.h"


BOOST_AUTO_TEST_CASE(test_relu) {
    std::cout << "Testing Relu" << std::endl;
    nn::Relu<float, 2> relu;

    int dim1 = 1;
    int dim2 = 10;
    Eigen::Tensor<float, 2> input(dim1, dim2);
    input.setRandom();
    input = input * input.constant(-1.0f);

    Eigen::Tensor<float, 2> result = relu.forward(input);
    for (unsigned int ii = 0; ii < dim1; ++ii) {
        for (unsigned int jj = 0; jj < dim2; ++jj) {
            BOOST_REQUIRE_MESSAGE(result(ii, jj) == 0, "Element in result does not equal zero");
        }
    }

    // Make a few elements positive
    input(0, 5) = 10.0;
    input(0, 3) = 150.0;

    result = relu.forward(input);
    for (unsigned int ii = 0; ii < dim1; ++ii) {
        for (unsigned int jj = 0; jj < dim2; ++jj) {
            BOOST_REQUIRE_MESSAGE(result(ii, jj) >= 0, "Element in result does is negative");
        }
    }
}

BOOST_AUTO_TEST_CASE(test_relu_back) {
    std::cout << "Testing Relu backwards" << std::endl;
    nn::Relu<float, 2> relu;

    int dim1 = 1;
    int dim2 = 10;
    Eigen::Tensor<float, 2> input(dim1, dim2);
    input.setValues({{-10, -7, -5, -3, 0, 1, 3, 5, 7, 10}});

    Eigen::Tensor<float, 2> accumulatedGrad(dim1, dim2);
    accumulatedGrad.setValues({{1, -4, 7, -10, 13, -16, 19, -22, 25, -28}});

    Eigen::Tensor<float, 2> forwardResult = relu.forward(input);
    Eigen::Tensor<float, 2> backwardResult = relu.backward(accumulatedGrad);

    std::vector<float> expectedOutput({0, 0, 0, 0, 0, -16, 19, -22, 25, -28});
    for (unsigned ii = 0; ii < dim2; ++ii) {
        BOOST_REQUIRE_MESSAGE(backwardResult(0, ii) == expectedOutput[ii], "Output of relu.backward did not match");
    }
}

BOOST_AUTO_TEST_CASE(test_softmax) {
    std::cout << "Testing Softmax" << std::endl;
    nn::Softmax<float, 2> softmax;

    int inputBatchSize = 2;
    Eigen::Tensor<float, 2> input(inputBatchSize, 2);
    input.setValues({{5, 5},
                     {-100, 100}});

    Eigen::Tensor<float, 2> result = softmax.forward(input);

    BOOST_REQUIRE_MESSAGE(result(0, 0) == 0.5, "Result(0, 0) did not match");
    BOOST_REQUIRE_MESSAGE(result(0, 1) == 0.5, "Result(0, 1) did not match");
    BOOST_REQUIRE_MESSAGE(result(1, 0) == 0, "Result (1, 0) did not match");
    BOOST_REQUIRE_MESSAGE(result(1, 1) == 1, "Result (1, 1) did not match");


    inputBatchSize = 1;
    int inputSize = 100;
    Eigen::Tensor<float, 2> input2(inputBatchSize, inputSize);
    input2.setRandom();

    Eigen::Tensor<float, 2> result2 = softmax.forward(input2);

    float sum = 0;
    for (unsigned int ii = 0; ii < inputSize; ++ii) {
        sum += result2(0, ii);
    }

    BOOST_REQUIRE_CLOSE(sum, 1.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(test_softmax_back) {
    std::cout << "Testing Softmax backwards" << std::endl;
    nn::Softmax<float, 2> softmax;

    int inputBatchSize = 2;
    Eigen::Tensor<float, 2> input(inputBatchSize, 2);
    input.setValues({{5, 7},
                     {-100, 100}});

    Eigen::Tensor<float, 2> result = softmax.forward(input);

    Eigen::Tensor<float, 2> labels(inputBatchSize, 2);
    labels.setValues({{0, 1},
                      {0, 1}});

    // Already has state from forward
    Eigen::Tensor<float, 2> backwardsResult = softmax.backward(labels);
    // TODO: Add actual tests. Test against TF/Pytorch?
}

BOOST_AUTO_TEST_CASE(test_net1) {
    std::cout << "Testing net creation" << std::endl;
    nn::Net<float> net;

    // TODO: output of previous should match input of next. Can we auto-infer in some nice way?
    int batchSize = 1;
    net.add(new nn::Dense<float, 2>(batchSize, 28 * 28, 100, true))
       .add(new nn::Dense<float, 2>(batchSize, 100, 100, true))
       .add(new nn::Dense<float, 2>(batchSize, 100, 10, true));

    Eigen::Tensor<float, 2> input(batchSize, 28 * 28);
    input.setRandom();
    Eigen::Tensor<float, 2> result = net.forward<2, 2>(input);
    BOOST_REQUIRE_MESSAGE(result.dimensions()[0] == batchSize, "Result dimension 0 did not match batch size");
    BOOST_REQUIRE_MESSAGE(result.dimensions()[1] == 10, "Result dimension 1 did not match last dense layer");
}

BOOST_AUTO_TEST_CASE(test_net2) {
    std::cout << "Testing net creation" << std::endl;
    nn::Net<> net;

    int batchSize = 64;
    int inputX = 28;
    int inputY = 28;
    int numClasses = 10;
    bool useBias = true;
    // Basic MLP for testing MNSIT
    net.add(new nn::Dense<>(batchSize, inputX * inputY, 100, useBias))
       .add(new nn::Relu<>())
       .add(new nn::Dense<>(batchSize, 100, 100, useBias))
       .add(new nn::Relu<>())
       .add(new nn::Dense<>(batchSize, 100, 10, useBias))
       .add(new nn::Relu<>())
       .add(new nn::Softmax<>());

    Eigen::Tensor<float, 2> input(batchSize, 28 * 28);
    input.setRandom();

    auto startTime = std::chrono::system_clock::now();
    Eigen::Tensor<float, 2> result = net.forward<2, 2>(input);
    auto endTime = std::chrono::system_clock::now();

    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "A single forward of size: [" << batchSize << ", 28, 28] took: " << duration.count() << "s"
              << std::endl;

    Eigen::Tensor<float, 2> fakeLabels(batchSize, numClasses);
    fakeLabels.setZero();
    fakeLabels.setValues({{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                          {0, 1, 0, 0, 0, 0, 0, 0, 0, 1}});

    net.backward<2>(fakeLabels);
}