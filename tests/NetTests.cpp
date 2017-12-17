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
    nn::Net<float> net;
    net.add(new nn::Dense<float, 2>({1, 1}, {1, 1}, true))
       .add(new nn::Dense<float, 2>({1, 1}, {1, 1}, true))
       .add(new nn::Dense<float, 2>({1, 1}, {1, 1}, true));
    net.printShapes();

    Eigen::Tensor<float, 2> input(1, 1);
    input.setRandom();

    std::cout << "Input: " << input << std::endl;

    Eigen::Tensor<float, 2> result = net.forward<2, 2>(input);

    std::cout << "Output: " << result << std::endl;
}