#include <vector>
#include <random>
#include <ctime>
#include <stdexcept>
#include "neuron.hpp"
#include "settings.h"


Neuron::Neuron (size_t numOfEdges, std::mt19937 &gen) : value(0), sum(0)
{
    weights.resize(numOfEdges);

    // generating random weights
    std::uniform_real_distribution<> generator(-2.4 / INPUT_SIZE, 2.4 / INPUT_SIZE);

    for (double &weight : weights)
        weight = generator(gen);
}


void Neuron::setWeight (size_t index, double value)
{
    if (index >= weights.size())
        throw std::out_of_range("Invalid weight index");

    weights[index] = value;
}


double Neuron::getWeight (size_t index) const
{
    if (index >= weights.size())
        throw std::out_of_range("Invalid weight index");

    return weights[index];
}


size_t Neuron::numberOfWeights() const
{
    return weights.size();
}


double& Neuron::operator[] (size_t index)
{
    return weights[index];
}
