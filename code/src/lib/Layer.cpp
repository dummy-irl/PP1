#include <Layer.h>

#include <MathUtils.h>

#include <cmath>
#include <iostream>
#include <utility>
#include <stdexcept>

namespace dnn {

Layer::Layer(uint64_t id, uint64_t seed, uint64_t layerInputSize, uint64_t layerOutputSize,
             ActivationFunction* activationFunction, std::string  name)
    : id(id), name(std::move(name)), seed(seed), layerInputSize(layerInputSize), layerOutputSize(layerOutputSize),
      activationFunction(activationFunction) {
    accumulatedDWeights.assign(layerOutputSize, std::vector<double>(layerInputSize, 0.0));
    accumulatedDBias.assign(layerOutputSize, 0.0);
}

Layer::~Layer() {
    delete activationFunction;
}

void Layer::init() {
    std::mt19937 randomGenerator(seed);

    // Glorot/Xavier uniform initialization:
    // scales the weight range by fan-in/fan-out so pre-activation variance stays roughly O(1)
    // regardless of layer width, instead of growing with layerInputSize the way a fixed
    // [-1, 1] range does.
    const double limit = std::sqrt(6.0 / static_cast<double>(layerInputSize + layerOutputSize));
    std::uniform_real_distribution<double> distribution(-limit, limit);

    weights.assign(layerOutputSize, std::vector<double>(layerInputSize));
    for (auto& row : weights) {
        for (double& value : row) {
            value = distribution(randomGenerator);
        }
    }

    // Zero-initialized bias
    bias.assign(layerOutputSize, 0.0);
}

void Layer::setParameters(std::vector<std::vector<double>> weights, std::vector<double> bias) {
    this->weights = std::move(weights);
    this->bias = std::move(bias);
}

void Layer::forwardPropagation(const std::vector<double>& previousLayerOutput) {
    // reset after each call
    preActivationOutput.assign(layerOutputSize, 0.0);
    activationOutput.assign(layerOutputSize, 0.0);
    for (std::uint64_t i = 0; i < layerOutputSize; i++) {
        // preActivationOutput[i] = weights[i] · previousLayerOutput + bias[i]
        preActivationOutput[i] = dnn::dotProduct(weights[i], previousLayerOutput) + bias[i];
        // activationOutput[i] = activation(preActivationOutput[i])
        activationOutput[i] = activationFunction->activation(preActivationOutput[i]);
    }
}

void Layer::backwardPropagation(const std::vector<double>& dActivationOutput, const std::vector<double>& previousLayerOutput) {
    
    if (!activationFunction) {throw std::logic_error("Layer has no activation function");}
    
    std::vector<double> dPreActivation(layerOutputSize);
    dWeights.resize(layerOutputSize);
    
    for (std::uint64_t i = 0; i < layerOutputSize; i++) {
        // dPreActivation[i] = dActivationOutput[i] · activationBackwards(preActivationOutput[i])
        dPreActivation[i] = dActivationOutput[i] * activationFunction->activationBackwards(preActivationOutput[i]);
        dWeights[i].resize(layerInputSize);
        for (std::uint64_t j = 0; j < layerInputSize; j++) {
            // dWeights[i][j] = dPreActivation[i] · previousLayerOutput[j]
            dWeights[i][j] = dPreActivation[i] * previousLayerOutput[j];
        }
    }
    dBias = dPreActivation;

    // accumulatedDWeights = accumulatedDWeights + dWeights
    for (std::size_t i = 0; i < dWeights.size(); i++) {
        for (std::size_t j = 0; j < dWeights[i].size(); j++) {
            accumulatedDWeights[i][j] += dWeights[i][j];
        }
    }

    // accumulatedDBias = accumulatedDBias + dBias
    for (std::size_t i = 0; i < dBias.size(); i++) {
        accumulatedDBias[i] += dBias[i];
    }
    numProcessedSamples++;
    dPreviousLayerOutput.assign(layerInputSize, 0.0);
    
    for (std::uint64_t j = 0; j < layerInputSize; j++) {
        std::vector<double> weightsColumn(layerOutputSize);
        for (std::uint64_t i = 0; i < layerOutputSize; i++) {
            // weightsColumn[i] = weights[i][j]
            weightsColumn[i] = weights[i][j];
        }
        // dPreviousLayerOutput[j] = weightsColumn · dPreActivation
        dPreviousLayerOutput[j] = dnn::dotProduct(weightsColumn, dPreActivation);
    }
}

void Layer::updateParameters(const std::vector<std::vector<double>>& dWeights, const std::vector<double>& dBias, double learningRate) {
    // weights = weights − learningRate · dWeights
    for (std::size_t i = 0; i < dWeights.size(); i++) {
        for (std::size_t j = 0; j < dWeights[i].size(); j++) {
            weights[i][j] -= learningRate * dWeights[i][j];
        }
    }
    // bias = bias − learningRate · dBias
    for (std::size_t i = 0; i < bias.size(); i++) {
        bias[i] -= learningRate * dBias[i];
    }
}

void Layer::commitToWeights(double learningRate) {
    // accumulatedDWeight = accumulatedDWeights/numProcessedSamples
    if (numProcessedSamples == 0) {return;}
    for (std::size_t i = 0; i < accumulatedDWeights.size(); i++) {
        for (std::size_t j = 0; j < accumulatedDWeights[i].size(); j++) {
            accumulatedDWeights[i][j] /= numProcessedSamples;
        }
    }
    // accumulatedDBias = accumulatedDBias/numProcessedSamples
    for (std::size_t i = 0; i< accumulatedDBias.size(); i++) {
        accumulatedDBias[i] /= numProcessedSamples;
    }

    Layer::updateParameters(accumulatedDWeights, accumulatedDBias, learningRate);

    for (auto& row : accumulatedDWeights) {
        row.assign(row.size(), 0.0);
    }
    accumulatedDBias.assign(accumulatedDBias.size(), 0.0);
    numProcessedSamples = 0;

}

uint64_t Layer::getId() const {
    return id;
}

const std::string& Layer::getName() const {
    return name;
}

const std::vector<double>& Layer::getActivationOutput() const {
    return activationOutput;
}

const std::vector<double>& Layer::getPreActivationOutput() const {
    return preActivationOutput;
}

ActivationFunction* Layer::getActivationFunction() const {
    return activationFunction;
}

const std::vector<std::vector<double>>& Layer::getWeights() const {
    return weights;
}

const std::vector<double>& Layer::getBias() const {
    return bias;
}

const std::vector<double>& Layer::getDPreviousLayerOutput() const {
    return dPreviousLayerOutput;
}

const std::vector<std::vector<double>>& Layer::getDWeights() const {
    return dWeights;
}

const std::vector<double>& Layer::getDBias() const {
    return dBias;
}

}
