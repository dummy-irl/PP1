#include <activation/LeakyReLU.h>

namespace dnn {

double LeakyReLU::activation(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0.01 * x;
    }
}

double LeakyReLU::activationBackwards(double x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 0.01;
    }
}

std::string LeakyReLU::getName() const {
    return "leaky_relu";
}

}