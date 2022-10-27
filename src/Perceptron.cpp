#include "Perceptron.hpp"

#include <cmath>
#include <stdexcept>
#include <iostream>

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

double Perceptron::get_output( const std::vector< double >& inputs )
{
    this-> __calc_activity( inputs );
    this-> __calc_activation();

    return this->activation;

}

double Perceptron::get_error( double desired_output ) const
{   
    return desired_output - this->activation;
}

double Perceptron::get_bigE( double desired_output ) const
{   
    return pow( this->get_error( desired_output ), 2 ) / 2;
}

void Perceptron::calc_delta( double desired_output )
{
    this->delta = ( this->get_error( desired_output ) ) * \
        this->__activation_function_ptr->derivative( this->activity );
}

void Perceptron::calc_delta_weights( const std::vector< double >& inputs )
{
    this->delta_weights.clear();
    for( size_t i = 0; i < inputs.size(); ++i )
        this->delta_weights.push_back( this->delta * inputs[ i ] );
}

void Perceptron::update_weights( bool include_bias )
{
    for( size_t i = 0; i < this->weights.size(); ++i )
        this->weights[ i ] += this->eta * this->delta_weights[ i ];

    if ( include_bias )
    {
        this->bias += this->eta * this->delta;
    }
}