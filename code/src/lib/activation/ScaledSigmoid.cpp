#include <activation/ScaledSigmoid.h>
#include <activation/Sigmoid.h>

namespace dnn {

ScaledSigmoid::ScaledSigmoid(double min, double max) : min(min), max(max) {}

double ScaledSigmoid::getMin() const {
    return min;
}

double ScaledSigmoid::getMax() const {
    return max;
}

double ScaledSigmoid::activation(double x) {
    Sigmoid sigmoid;
    double sig = sigmoid.activation(x);
    return min + (max - min) * sig;
}

double ScaledSigmoid::activationBackwards(double x) {
    Sigmoid sigmoid;
    double sig = sigmoid.activation(x);
    return (max - min) * sig * (1.0 - sig);
}

std::string ScaledSigmoid::getName() const {
    return "scaled_sigmoid";
}

}