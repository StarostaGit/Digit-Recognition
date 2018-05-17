#ifndef LAYER_HPP
#define LAYER_HPP

#include "neuron.hpp"


class Layer
{
    public:

        // An enum defining a type of the layer
        enum Type {INPUT, HIDDEN, OUTPUT};

        // Type of the layer
        Type type;

        // A bias value for a given layer
        double bias;

        // Constructor creating a new Layer object with a specified
        // number of neurons
        Layer (size_t size, Type type, size_t nextLayerSize, std::mt19937 &gen);

        // Setters and getters
        size_t getSize () const;
        size_t getNextLayerSize () const;

        // Operator [] allows access and modification of neurons
        // in a direct manner
        Neuron& operator[] (size_t index);

    private:

        // Neurons in next layer
        size_t nextLayerSize;

        // Container for underlying neurons
        std::vector<Neuron> neurons;

};


#endif // LAYER_HPP
