### nn_cpp - Minimalistic C++11 header only Neural Network Library

Currently under active development, with several minimally viable product features missing (see Todo subsection below)

We make heavy use of the [Eigen::Tensor](https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md) library for computation.

The goal is to make a simple library to inline small neural networks in modern C++ code without the overhead of a major framework.

### Usage Example

```c++

nn::Net<float> net;
net.add(new nn::Dense<float, 2>(batchSize, inputSize, numHiddenNodes, useBias));
net.add(new nn::Relu<float, 2>());
net.add(new nn::Dense<float, 2>(batchSize, numHiddenNodes, outputSize, useBias));
net.add(new nn::Softmax<float, 2>());
 
nn::CrossEntropyLoss<float, 2> lossFunc;
net.registerOptimizer(new nn::Adam<float>(learningRate));
 
int numEpoch = 100;
for (int currentEpoch = 0; currentEpoch < numEpoch; ++currentEpoch) {
    auto result = net.forward<2, 2>(input);
    auto loss = lossFunc.loss(result, labels);
    
    net.backward(lossFunc.backward(result, labels));
    net.step();
    
    std::cout << "Epoch: " << currentEpoch << " Loss: " << loss << std::endl;
}
```

### Dependencies

```
Eigen 3.3.4
Boost
```

### Todo
Currently a ton of basic functionality that makes it worth using is missing

- Saving and loading of weights
- Support multi-core CPU usage, and GPU usage
- Additional activation layers
- Additional loss layers
- Convolutions
- Alternatives to gradient descent (SGD, Adam, etc)
- Memory optimizations when test mode only
- Add proper logging


### Building
Assuming you have the above dependencies installed there is nothing to build if you simply want to use the library. Simply include nn/Net.h

If you want to run the tests, do the following

```bash
mkdir build
cd build
cmake -D NNCPP_BUILD_TESTS=ON ..
make test
```

### Examples
Currently, there are two very simple examples of training an MLP on some data.

[BasicMLP.cpp](./examples/BasicMLP.cpp) - A two class problem of classifying separable 2D uniformly distributed data (donut vs donut hole)

[IrisTest.cpp](./examples/IrisTest.cpp) - Classifying the Iris dataset with a simple MLP.

These are built if you do the following:

```bash
mkdir build
cd build
cmake -D NNCPP_BUILD_EXAMPLES=ON ..
make
```

To run, simply run them via the binary.

```bash
cd build
./bin/iris_test

# Or, for the basic mlp test

./bin/basic_mlp
```