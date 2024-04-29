#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.141592653589793238463
#endif

const int TRAINING_SET_SIZE = 20;
const int NEURONS = 5;
const double EPSILON = 0.05;
const int EPOCH = 50000;

double neuronScalarOffset[NEURONS] = {}; // "C" in the writing
double weight[NEURONS] = {}; // "W" in the writing
double V[NEURONS] = {};
double b = 0;

//for clarity
typedef double angle;
typedef double sinValue;

//this behaves like a soft clamp, clamping values between 0 and 1 with a tighter gradient between -1 and 1 and smoother gradient near the asymptotes (-inf -> 0, inf -> 1)
//Output will always be between 0 and 1
double sigmoid(double x){
    return (1.0f / (1.0f + std::exp(-x)));
}

//this is called both during training and when prompting the model for a result, so this appears to be where we query the model
//Variable behaviors re:outputs look like this:
//Value increases with total values of all V, neuronScalarOffset, and weight, with "all V" having the greatest effect
double f_theta(double x){
    double result = b;

    //For every neuron: calculate a value which consists of that neuron's "scalar offset" plus its weight, multiplied by the function input, projected to a sigmoid, multiplied by this neuron's "V", and add it to the result value, then return the result value
    for (int i = 0; i < NEURONS; i++){
        result += V[i] * sigmoid(neuronScalarOffset[i] + weight[i] * x);
    }

    return result;
}

//'x' is a training input, 'y' is a training output (ie, "for the value x, we should get y")
void train(angle x, sinValue y){
    for (int i = 0; i < NEURONS; i++){
        weight[i] =
            weight[i] -
                EPSILON * 2 * (f_theta((angle)x) - (sinValue)y)
                * V[i] * (angle)x
                * (1 - sigmoid(neuronScalarOffset[i] + weight[i] * (angle)x))
                * sigmoid(neuronScalarOffset[i] + weight[i] * (angle)x);
    }

    for (int i = 0; i < NEURONS; i++){
        V[i] = V[i] - EPSILON * 2 * (f_theta(x) - y) * sigmoid(neuronScalarOffset[i] + weight[i] * x);
    }

    b = b - EPSILON * 2 * (f_theta(x) - y);
}

int main() {
    std::vector<std::pair<angle, sinValue>> trainingSet;
    trainingSet.resize(TRAINING_SET_SIZE);

    srand(time(NULL));

    for (int i = 0; i < NEURONS; i++){
        /*This feels like an implementation bug:
         * The apparent intention is that the values are populated with random vals between -1 and 1.
         * However without explicitly declaring the constants as floats, this appears to always generate -1
         * This behavior was apparent on several C++ standards; none of the handful I tried did what I assume is intended
         * For the time being, we'll leave it as the author wrote it
         * tl;dr: all values of weight, V, and neuronScalarOffset will = -1 after this block
         */
        weight[i] = 2 * rand() / RAND_MAX - 1;
        V[i] = 2 * rand() / RAND_MAX - 1;
        neuronScalarOffset[i] = 2 * rand() / RAND_MAX - 1;
    }

    //Training set is formatted as ("an increment of 2pi radians", "the sine of that angle")
    for (int i = 0; i < TRAINING_SET_SIZE; i++){
        trainingSet[i] = std::make_pair((angle)(i * 2 * M_PI / TRAINING_SET_SIZE), (sinValue)(std::sin(i * 2 * M_PI / TRAINING_SET_SIZE)));
    }

    //Train the model EPOCH times
    for (int i = 0; i < EPOCH; i++){
        for (int j = 0; j < TRAINING_SET_SIZE; j++){
            train((angle)trainingSet[i].first, (sinValue)trainingSet[i].second);
        }

        std::cout << i << "\r";
    }
}
