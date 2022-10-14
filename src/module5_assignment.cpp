#include <iostream>
#include <iomanip>

#include "Perceptron.hpp"

int main()
{
    Perceptron perceptron( { 0.24, 0.88 }, 0.0, 5.0, "Sigmoid" );

    int max_iters = 100;
    std::vector< double > inputs = { 0.8, 0.9 };
    double desired_output = 0.95;
    std::cout << std::setprecision(4) << std::fixed;

    std::cout << "\n-------------------------------------------------\n";
    std::cout << "Question1" << "\n-------------------------------------------------\n";
    for ( size_t i = 0; i < max_iters; ++i )
    {
        perceptron.get_output( inputs );
        perceptron.calc_delta( desired_output );
        perceptron.calc_delta_weights( inputs );
        perceptron.update_weights( );
        std::cout << "Iteration: " << i << "\tActivation: " << perceptron.activation
            << "\tw1: " << perceptron.weights[ 0 ]
            << "\tw2: " << perceptron.weights[ 1 ]
            << "\tdelta: " << perceptron.delta  << "\n";
    }

    std::cout << "\n-------------------------------------------------\n";


    std::cout << "Question2" << "\n-------------------------------------------------\n";
    perceptron = Perceptron( { 0.24, 0.88 }, 0.0, 5.0, "Sigmoid" );
    desired_output = 0.15;

    for ( size_t i = 0; i < max_iters; ++i )
    {
        perceptron.get_output( inputs );
        perceptron.calc_delta( desired_output );
        perceptron.calc_delta_weights( inputs );
        perceptron.update_weights( );
        std::cout << "Iteration: " << i << "\tActivation: " << perceptron.activation
            << "\tw1: " << perceptron.weights[ 0 ]
            << "\tw2: " << perceptron.weights[ 1 ]
            << "\tdelta: " << perceptron.delta  << "\n";
    }

    std::cout << "\n-------------------------------------------------\n";


}

