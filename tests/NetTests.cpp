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
#include "Net.h"


BOOST_AUTO_TEST_CASE(test_net) {
    std::cout << "Testing net creation" << std::endl;
    nn::Net<float> net;

    // TODO: output of previous should match input of next. Can we auto-infer in some nice way?
    int batchSize = 1;
    net.add(new nn::Dense<float, 2>(batchSize, 28 * 28, 100, true))
       .add(new nn::Dense<float, 2>(batchSize, 100, 100, true))
       .add(new nn::Dense<float, 2>(batchSize, 100, 10, true));

    Eigen::Tensor<float, 2> input(batchSize, 28 * 28);
    input.setRandom();

    std::cout << "Input: " << input << std::endl;
    Eigen::Tensor<float, 2> result = net.forward<2, 2>(input);
    std::cout << "Output: " << result << std::endl;
}

BOOST_AUTO_TEST_CASE(test_relu) {
    std::cout << "Testing Relu" << std::endl;
    nn::Net<float> net;

    net.add(new nn::Relu<float, 2>());

    int dim1 = 1;
    int dim2 = 10;
    Eigen::Tensor<float, 2> input(dim1, dim2);
    input.setRandom();
    input = input * input.constant(-1.0f);

    Eigen::Tensor<float, 2> result = net.forward<2, 2>(input);
    for (unsigned int ii = 0; ii < dim1; ++ii) {
        for (unsigned int jj = 0; jj < dim2; ++jj) {
            BOOST_REQUIRE_MESSAGE(result(ii, jj) == 0, "Element in result does not equal zero");
        }
    }

    // Make a few elements positive
    input(0, 5) = 10.0;
    input(0, 3) = 150.0;

    result = net.forward<2, 2>(input);
    for (unsigned int ii = 0; ii < dim1; ++ii) {
        for (unsigned int jj = 0; jj < dim2; ++jj) {
            BOOST_REQUIRE_MESSAGE(result(ii, jj) >= 0, "Element in result does is negative");
        }
    }
}