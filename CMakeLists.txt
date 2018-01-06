cmake_minimum_required(VERSION 3.0)
project(nn_cpp CXX)
include_directories(nn)

set(CMAKE_CXX_STANDARD 11)
option(NNCPP_BUILD_TESTS "Whether to build the tests" ON)
option(NNCPP_BUILD_EXAMPLES "Whether to build examples" ON)

file(GLOB_RECURSE nn_sources nn/*.h)
add_library(nn_cpp ${nn_sources})
set_target_properties(nn_cpp PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(nn_cpp PUBLIC nn)

find_package(Eigen3 REQUIRED)
target_include_directories(nn_cpp PUBLIC ${EIGEN3_INCLUDE_DIR})


if (NNCPP_BUILD_TESTS)
    find_package(Boost COMPONENTS unit_test_framework)

    if (Boost_FOUND)
        enable_testing()

        add_executable(neural_net_test tests/NetTests.cpp)
        target_link_libraries(neural_net_test ${Boost_LIBRARIES} nn_cpp)
        add_test(NAME neural_net_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND neural_net_test)

        add_executable(loss_test tests/LossTests.cpp)
        target_link_libraries(loss_test ${Boost_LIBRARIES} nn_cpp)
        add_test(NAME loss_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND loss_test)
    endif()
endif()

if (NNCPP_BUILD_EXAMPLES)

    add_executable(basic_mlp examples/BasicMLP.cpp)
    target_link_libraries(basic_mlp nn_cpp)

    add_executable(iris_test examples/IrisTest.cpp)
    target_link_libraries(iris_test nn_cpp)
endif()