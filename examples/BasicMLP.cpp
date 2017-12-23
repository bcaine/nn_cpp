/**
 * @file BasicMLP.cpp
 *
 * @breif A very basic MLP to test training a network
 *
 * @date 12/22/17
 * @author Ben Caine
 */
#include "../nn/Net.h"
#include <random>
#include <vector>

struct ToyLabeledData {
    std::vector<std::pair<float, float>> data;
    std::vector<int> labels;

    int getSize() {
        assert (labels.size() == data.size());
        return labels.size();
    }
};

ToyLabeledData generateCircleData(int numInnerPoints, int numOuterPoints) {
    float innerRadius = 1.0;
    float outerRadius = 3.0;
    int numTotalPoints= numInnerPoints + numOuterPoints;

    ToyLabeledData dataset;

    // Random generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> uniformDist(-innerRadius, innerRadius);

    // Create inner circle data (class 0)
    while (dataset.data.size() < numInnerPoints) {
        auto x = static_cast<float>(uniformDist(rng));
        auto y = static_cast<float>(uniformDist(rng));
        dataset.data.emplace_back(std::pair<float, float>{x, y});
        dataset.labels.push_back(0);
    }

    // Update uniform dist to be from -outer to outer
    uniformDist = std::uniform_real_distribution<>(-outerRadius, outerRadius);
    auto getOuterVal = [&]() {
        float val = 0;
        // Reject ones inside the inner circle
        while (std::abs(val) < innerRadius) {
            val = static_cast<float>(uniformDist(rng));
        }
        return val;
    };

    // Fill the rest with outer circle points
    while (dataset.data.size() < numTotalPoints) {
        float x = getOuterVal();
        float y = getOuterVal();

        dataset.data.emplace_back(std::pair<float, float>{x, y});
        dataset.labels.push_back(1);
    }

    return dataset;
};

int main() {
    nn::Net<float> net;

    int firstClassSize = 50;
    int secondClassSize = 50;
    int batchSize = firstClassSize + secondClassSize;
    int inputSize = 2;
    int numClasses = 2;
    bool useBias = true;

    Eigen::Tensor<float, 2> inputData(batchSize, inputSize);
    Eigen::Tensor<float, 2> labels(batchSize, inputSize);
    inputData.setZero();
    labels.setZero();

    auto dataset = generateCircleData(firstClassSize, secondClassSize);
    int datasetSize = dataset.getSize();
    for (unsigned int ii = 0; ii < datasetSize; ++ii) {
        inputData(ii, 0) = dataset.data[ii].first;
        inputData(ii, 1) = dataset.data[ii].second;
        // Set up one hot encoding
        labels(ii, 0) = static_cast<float>(dataset.labels[ii] == 0);
        labels(ii, 1) = static_cast<float>(dataset.labels[ii] == 1);
    }

    net.add(new nn::Dense<>(batchSize, inputSize, 6, useBias));
    net.add(new nn::Relu<>());
    net.add(new nn::Dense<>(batchSize, 6, 2, useBias));
    net.add(new nn::Relu<>());
    net.add(new nn::Softmax<>());

    auto result = net.forward<2, 2>(inputData);
    std::cout << result << std::endl;

    return 0;
}

