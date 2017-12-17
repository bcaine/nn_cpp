### nn_cpp (Currently non-functional)
nn_cpp is a header only minimalistic neural network library with Eigen (and boost unit tests for testing) as its only dependency.

We make heavy use of the [Eigen::Tensor](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md) library for computation.

The goal is to make a simple library to inline small neural networks in modern C++ code without the overhead of a major framework.

### Requirements

```
Eigen 3.3.4
Boost unit_test_framework
```

### Building
Assuming you have the above dependencies installed there is nothing to build if you simply want to use the library. Simply include nn/Net.h

If you want to run the tests, do the following

```bash
mkdir build
cd build
cmake ..
make test
```