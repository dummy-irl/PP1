#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);

// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);

template <typename T>
std::vector<double> normalize(const std::vector<T>& values) {
    std::vector<double> result;
    if (values.empty()) {
        return result;
    }

    const double maxValue = static_cast<double>(*std::max_element(values.begin(), values.end()));
    if (maxValue == 0) {
        throw std::invalid_argument("cannot normalize with a zero maximum");
    }

    result.reserve(values.size());
    for (const auto& value : values) {
        result.push_back(static_cast<double>(value) / maxValue);
    }
    return result;
}

template <typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue) {
    std::vector<double> result;
    if (values.empty()) {
        return result;
    }
    if (maxValue == 0) {
        throw std::invalid_argument("cannot normalize with a zero maximum");
    }

    result.reserve(values.size());
    for (const auto& value : values) {
        result.push_back(static_cast<double>(value) / maxValue);
    }
    return result;
}

}
