#include <activation/ReLU.h>

namespace dnn {

double ReLU::activation(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0.0;
    }
}

double ReLU::activationBackwards(double x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

std::string ReLU::getName() const {
    return "relu";
}

}
