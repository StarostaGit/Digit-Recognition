#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cstddef>
#include <random>


class Neuron
{
    public:

        // Current value of the neuron and the sum before applying
        // the activation function
        double value;
        double sum;

        // Constructor creating a new Neuron object with a specified
        // number of outgoing weights and a random value
        Neuron (size_t numOfEdges, std::mt19937 &gen);

        // Getters and setters
        void setWeight (size_t index, double value);
        double getWeight (size_t index) const;
        size_t numberOfWeights () const;

        // Operator [] allows access and modification of weights
        // container in a direct manner
        double& operator[] (size_t index);

    private:

        // Container for outgoing weights
        std::vector<double> weights;

};


#endif // NEURON_HPP
