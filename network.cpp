#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <cstddef>
#include "network.hpp"


Network::Network (size_t numOfInputNeurons, size_t numOfHiddenLayers, size_t neuronsInHiddenLayer,
                  size_t numOfOutputNeurons, Function &activationFunction, double learningRate)
                  : debug(false),
                    inputSize(numOfInputNeurons),
                    numOfHiddenLayers(numOfHiddenLayers),
                    hiddenLayerSize(neuronsInHiddenLayer),
                    categories(numOfOutputNeurons),
                    learningRate(learningRate),
                    activationFunction(activationFunction)
{
    // Building the structure of the network
    std::mt19937 gen(time(0));
    layers.push_back( Layer(numOfInputNeurons, Layer::INPUT, hiddenLayerSize, gen) );

    for (int i = 1; i < numOfHiddenLayers; i++)
        layers.push_back( Layer(neuronsInHiddenLayer, Layer::HIDDEN, hiddenLayerSize, gen) );
    if (numOfHiddenLayers)
        layers.push_back( Layer(neuronsInHiddenLayer, Layer::HIDDEN, categories, gen) );

    layers.push_back( Layer(numOfOutputNeurons, Layer::OUTPUT, 0, gen) );
}


std::vector <double> Network::feedForward (std::vector <double> input)
{
    // Checking for correctness
    if (input.size() != inputSize)
        throw std::invalid_argument("Input vector's size does not match");

    // Computing first layer
    for (int i = 0; i < inputSize; i++)
        layers[0][i].value = input[i];

    input.resize(layers[0].getNextLayerSize());
    for (double &d : input)
        d = 0;

    // Iterating through all neurons of the input layer
    for (int i = 0; i < inputSize; i++)
    {
        for (int j = 0; j < layers[0][i].numberOfWeights(); j++)
            input[j] += layers[0][i].value * layers[0][i][j];
    }

    // Adding the bias
    for (double &d : input)
        d += layers[0].bias;

    // Feed forward
    for (auto it = layers.begin() + 1; it < layers.end(); it++)
    {
        Layer &layer = *it;

        // Updating current values
        for (int i = 0; i < layer.getSize(); i++)
        {
            layer[i].sum = input[i];
            layer[i].value = activationFunction(input[i]);
        }

        // Reseting temporary container for accumulation
        input.resize(layer.getNextLayerSize());
        for (double &d : input)
            d = 0;

        // Iterating through all neurons
        for (int i = 0; i < layer.getSize(); i++)
        {
            // Iterating through all weights
            for (int j = 0; j < layer[i].numberOfWeights(); j++)
                input[j] += (layer[i].value * layer[i][j]);
        }

        // Adding the bias
        for (double &d : input)
            d += layer.bias;
    }

    // Copying the output
    input.resize(categories);
    for (int i = 0; i < categories; i++)
        input[i] = layers[numOfHiddenLayers + 1][i].value;


    // PRINTING
//    std::cout << "-------- NETWORK --------" << std::endl;
//    std::cout << "Neuron 1 : " << layers[0][0].value << std::endl;
//    std::cout << "Weight 1 : " << layers[0][0][0] << std::endl;
//    std::cout << "Bias 1 : " << layers[0].bias << std::endl;
//    std::cout << "Neuron 2 sum : " << layers[1][0].sum << std::endl;
//    std::cout << "Neuron 2 : " << layers[1][0].value << std::endl;
//    std::cout << "Weight 2 : " << layers[1][0][0] << std::endl;
//    std::cout << "Bias 2 : " << layers[1].bias << std::endl;
//    std::cout << "Neuron 3 sum : " << layers[2][0].sum << std::endl;
//    std::cout << "Neuron 3 : " << layers[2][0].value << std::endl;
//    std::cout << "Bias 3 : " << layers[2].bias << std::endl;



    return input;
}


int Network::guess (std::vector <double> input)
{
    input = feedForward(input);

    size_t best = 0;
    for (int i = 0; i < input.size(); i++)
        best = input[best] < input[i] ? i : best;

    return best;
}


void Network::backpropagation (const std::vector <double> &output)
{
    std::vector<double> accumulator, newSums;
    double temp, gradient, biasGradient;

    // Checking for correctness
    if (output.size() != categories)
        throw std::invalid_argument("Output vector's size does not match");

    // Calculating errors
    accumulator.resize(categories);
    for (int i = 0; i < categories; i++)
        accumulator[i] = layers[numOfHiddenLayers + 1][i].value - output[i];

    // Diagnostic information, only if debug flag set to true
    if (debug)
    {
        temp = 0;
        for (double &d : accumulator)
            temp += (d * d);
        std::cout << "Output: " << layers.back()[0].value << std::endl;
        std::cout << "Error: " << 0.5 * temp << std::endl;
    }

    // Propagating the error backwards
    for (auto it = layers.end() - 2; it >= layers.begin(); it--)
    {
        Layer &layer = *it;
        newSums.resize(layer.getSize());

        for (double &sum : newSums)
            sum = 0;
        biasGradient = 0;

        for (int i = 0; i < layer.getSize(); i++)
        {
            // Updating weights
            for (int j = 0; j < layer[i].numberOfWeights(); j++)
            {
                temp = activationFunction.derivative( (*(it + 1))[j].sum ) * accumulator[j];
                biasGradient += temp;
                newSums[i] += temp * layer[i][j];
                gradient = temp * layer[i].value;
                //std::cout << "Gradient: " << gradient << std::endl;
                layer[i][j] -= learningRate * gradient;
            }
        }

        // Updating the bias
        layer.bias -= learningRate * biasGradient;
        //std::cout << "Bias gradient: " << biasGradient << std::endl;

        accumulator = newSums;
    }
}


void Network::setDebugMode (bool flag)
{
    debug = flag;
}
