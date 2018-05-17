#include <random>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <windows.h>
#include <stdexcept>
#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"
#include "network.hpp"
#include "settings.h"


int main()
{
    Sigmoid sigmoid;
    std::vector< std::vector<double> > trainingData, testData, trainingOutput;
    std::vector<int> testLabels;
    double target;

    Network digitRecognizer = Network(INPUT_SIZE, HIDDEN_LAYERS, HIDDEN_LAYER_NEURONS,
                                      OUTPUT_SIZE, sigmoid, ALPHA);

    digitRecognizer.setDebugMode(true);

    std::cout << std::fixed << std::setprecision(5);

    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

    // Abort if dataset is unavailable
    if (dataset.training_images.empty())
        throw std::runtime_error("Couldn't load the dataset");

    //mnist::normalize_dataset(dataset);

    // Recasting to decimal types
    for (int i = 0; i < DATASET_SIZE; i++)
    {
        trainingData.push_back( std::vector<double>() );
        trainingOutput.push_back( std::vector<double>() );

        for (int j = 0; j < INPUT_SIZE; j++)
            trainingData[i].push_back( double(dataset.training_images[i][j]) / 255 );

        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            if (dataset.training_labels[i] == j)
                trainingOutput[i].push_back(1.0);
            else
                trainingOutput[i].push_back(0.0);
        }
    }

    for (int i = 0; i < TEST_DATA_SIZE; i++)
    {
        testLabels.push_back( int(dataset.test_labels[i]) );
        testData.push_back( std::vector<double>() );

        for (int j = 0; j < INPUT_SIZE; j++)
            testData[i].push_back( double(dataset.test_images[i][j]) / 255 );
    }

    // Training
    /*std::vector<bool> done;
    for (int i = 0; i < DATASET_SIZE; i++)
        done.push_back(false);
    srand(time(0));
    int index;
    for (int i = 0; i < DATASET_SIZE; i+=10)
    {
        index = rand() % DATASET_SIZE;
        while (done[index])
            index = rand() % DATASET_SIZE;
        auto output = digitRecognizer.feedForward(trainingData[index]);
        for (auto &obj : output)
            std::cout << obj << std::endl;
        digitRecognizer.backpropagation(trainingOutput[index]);
        done[index] = 1;
    }*/

    srand(time(0));
    for (int i = 0; i < 100000; i++)
    {
        target = double(rand()) / RAND_MAX;
        std::cout << std::endl << target << std::endl;
        digitRecognizer.feedForward({target});
        digitRecognizer.backpropagation({target});
    }

    return 0;
}
