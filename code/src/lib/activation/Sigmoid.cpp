#include <activation/Sigmoid.h>
#include <cmath>

namespace dnn {

double Sigmoid::activation(double x) {
    double exp = std::exp(-x);
    return 1.0/(1.0 + exp);
}

double Sigmoid::activationBackwards(double x) {
    double act = activation(x);
    return act * (1.0 - act);
}

std::string Sigmoid::getName() const {
    return "sigmoid";
}

}