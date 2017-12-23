cmake_minimum_required(VERSION 3.0)
project(nn_cpp CXX)
include_directories(nn)

set(CMAKE_CXX_STANDARD 11)
option(NNCPP_BUILD_TESTS "Whether to build the tests" ON)
option(NNCPP_BUILD_EXAMPLES "Whether to build examples" ON)

file(GLOB_RECURSE nn_sources nn/*.h)
add_library(neural_net ${nn_sources})
set_target_properties(neural_net PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(neural_net PUBLIC nn)

find_package(Eigen3 REQUIRED)
target_include_directories(neural_net PUBLIC ${EIGEN3_INCLUDE_DIR})


if (NNCPP_BUILD_TESTS)
    find_package(Boost COMPONENTS unit_test_framework)

    if (Boost_FOUND)
        enable_testing()

        add_executable(neural_net_test tests/NetTests.cpp)
        target_link_libraries(neural_net_test ${Boost_LIBRARIES} neural_net)
        add_test(NAME neural_net_test WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND neural_net_test)
    endif()
endif()

if (NNCPP_BUILD_EXAMPLES)
    add_executable(basic_mlp examples/BasicMLP.cpp)
    target_link_libraries(basic_mlp neural_net)
endif()