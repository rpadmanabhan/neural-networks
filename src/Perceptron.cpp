#include "Perceptron.hpp"

#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>


struct SigmoidActivation: ActivationFunction
{

    virtual double value( double x ) const
    {
        return 1 / (1 + exp( -1 * x ) );
    }

    virtual double derivative( double x ) const
    {
        double val = this->value( x );
        return ( 1 - val ) * val;
    }
};


Perceptron::Perceptron(
    const std::vector< double >& weights,
    double bias,
    double eta,
    const std::string& activation_funtion
    )
    {
        // Set the appropriate activation function
        if ( activation_funtion == "Sigmoid" )
            this->__activation_function_ptr = std::make_unique< SigmoidActivation >();
        else
            throw std::invalid_argument( activation_funtion + " is not supported !" );

        this->weights = weights;
        this->bias    = bias;
        this->eta     = eta;
    }

void Perceptron::__calc_delta( double desired_output )
{
    this->delta = ( desired_output - this->activation ) * \
        this->__activation_function_ptr->derivative( this->activation );
}


void Perceptron::__calc_activity( const std::vector< double >& inputs )
{
    // activity = bias + sum[inputs_i * weights_i]
    this->activity = this->bias;
    for ( size_t i = 0; i < inputs.size(); ++i )
    {
        this->activity += inputs[ i ] * this->weights[ i ];
    }
}

void Perceptron::__calc_activation()
{
    this->activation = this->__activation_function_ptr->value( this->activity );
}

void Perceptron::__calc_delta_weights( const std::vector< double >& inputs, double desired_output )
{
    this->delta_weights.clear();
    this->__calc_delta( desired_output );
    for( size_t i = 0; i < inputs.size(); ++i )
        this->delta_weights.push_back( this->delta * inputs[ i ] );
}

void Perceptron::update_weights( const std::vector< double >& inputs, double desired_output )
{
    this->__calc_activity( inputs );
    this->__calc_activation();
    this->__calc_delta_weights( inputs, desired_output );
    for( size_t i = 0; i < this->weights.size(); ++i )
        this->weights[ i ] += this->eta * this->delta_weights[ i ];

    this->bias += this->eta * this->delta;
}

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
        perceptron.update_weights( inputs, desired_output );
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
        perceptron.update_weights( inputs, desired_output );
        std::cout << "Iteration: " << i << "\tActivation: " << perceptron.activation
            << "\tw1: " << perceptron.weights[ 0 ]
            << "\tw2: " << perceptron.weights[ 1 ]
            << "\tdelta: " << perceptron.delta  << "\n";
    }

    std::cout << "\n-------------------------------------------------\n";


}

