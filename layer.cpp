#include <cstddef>
#include "layer.hpp"


Layer::Layer (size_t size, Layer::Type type, size_t nextLayerSize, std::mt19937 &gen)
                : type(type), bias(0), nextLayerSize(nextLayerSize)
{
    while (size--)
        neurons.push_back(Neuron(nextLayerSize, gen));
}


size_t Layer::getSize () const
{
    return neurons.size();
}


size_t Layer::getNextLayerSize () const
{
    return nextLayerSize;
}


Neuron& Layer::operator[] (size_t index)
{
    return neurons[index];
}
