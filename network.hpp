#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <cstddef>
#include "functions.hpp"
#include "layer.hpp"


class Network
{

    public:

        // Constructor of a fully connected network with specified number of hidden layers,
        // number of neurons in those layers, number of input neurons and number of output neurons.
        // It also takes an object which implements the Function interface, using it as an activation
        // function and a learning rate multiplier.
        Network (size_t numOfInputNeurons, size_t numOfHiddenLayers, size_t neuronsInHiddenLayer,
                 size_t numOfOutputNeurons, Function &activationFunction, double learningRate);

        // Method performing backpropagation on a given set of outputs
        void backpropagation (const std::vector<double> &output);

        // Method feeding forward an input and returning an output vector
        std::vector<double> feedForward (std::vector<double> input);

        // Method returning a highest ranking category computed from a given input
        int guess (std::vector<double> input);

        // Method turning the debug mode on/off
        void setDebugMode (bool flag);


    private:

        // Parameters
        bool debug;
        size_t inputSize;
        size_t numOfHiddenLayers;
        size_t hiddenLayerSize;
        size_t categories;
        double learningRate;
        Function &activationFunction;

        // Neural Network's body
        std::vector <Layer> layers;

};


#endif // NETWORK_HPP
