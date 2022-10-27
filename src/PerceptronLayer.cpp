#include "PerceptronLayer.hpp"


PerceptronLayer::PerceptronLayer(
    size_t num_perceptrons,
    const std::vector< std::vector< double > >& initial_weights,
    const std::vector< double >& initial_biases,
    double eta,
    bool is_output_layer,
    const std::string& activation_funtion )
{
    for ( size_t i = 0; i < num_perceptrons; ++i )
    {
        this->perceptrons.emplace_back( Perceptron( initial_weights[ i ], initial_biases[ i ], eta, activation_funtion ) );
    }

    this->previous_weights = initial_weights;
    this->previous_biases = initial_biases;
    this->weights = initial_weights;
    this->biases = initial_biases;
    this->is_output_layer = is_output_layer;
    this->eta = eta;
    this->activation_function = activation_funtion;
}

void PerceptronLayer::calc_error(
    const std::vector< double >& desired_output
)
{
    this->error.clear();
    size_t i = 0;
    for ( auto& perceptron : this->perceptrons )
    {
        this->error.push_back( perceptron.get_error( desired_output[i] ) );
        ++i;
    }
}

double PerceptronLayer::get_layer_bigE( const std::vector< double >& desired_output )
{
    double bigE = 0;
    size_t i = 0;
    for ( auto& perceptron : this->perceptrons )
    {
        bigE += perceptron.get_bigE( desired_output[ i ] );
        ++i;
    }
    return bigE;
}


void PerceptronLayer::calc_layer_output(
    const std::vector< double >& inputs
)
{
    this->output.clear();
    size_t i = 0;
    for ( auto& perceptron : this->perceptrons )
    {
        this->output.push_back( perceptron.get_output( inputs ) );
        ++i;
    }
}

size_t PerceptronLayer::get_layer_length(
)
{
    return this->perceptrons.size();
}

void PerceptronLayer::calc_layer_delta_values(
    const std::vector< double >& desired_output
)
{
    size_t i = 0;
    for ( auto& perceptron : this->perceptrons )
    {
        perceptron.calc_delta( desired_output[ i ] );
    }
}

void PerceptronLayer::calc_layer_delta_values(
    const PerceptronLayer& above_layer
)
{
    size_t i = 0;
    for ( auto& this_perceptron : this->perceptrons )
    {
        double sum = 0;
        size_t j = 0;
        for ( auto& perceptron : above_layer.perceptrons )
        {
            sum += perceptron.delta * above_layer.previous_weights[j][ i ];
            ++j;
        }
        ++i;
        this_perceptron.delta = (1 - this_perceptron.activation) * this_perceptron.activation * sum;
    }
}

void PerceptronLayer::calc_layer_delta_weights(
    const std::vector< double >& inputs 
)
{
    for ( auto& perceptron: this->perceptrons )
    {
        perceptron.calc_delta_weights( inputs );
    }
}

void PerceptronLayer::update_layer_weights( bool include_bias )
{
    this->previous_weights = this->weights;
    this->previous_biases = this->biases;
    this->biases.clear();
    this->weights.clear();
    for ( auto& perceptron: this->perceptrons )
    {
        perceptron.update_weights( include_bias );
        this->weights.push_back( perceptron.weights );
    }
}

void PerceptronLayer::update( const std::vector< double >& inputs, const std::vector< double >& desired_output, bool include_bias )
{
    this->calc_layer_delta_values( desired_output );
    this->calc_layer_delta_weights( inputs );
    this->update_layer_weights( include_bias );
}

void PerceptronLayer::update( const std::vector< double >& inputs, const PerceptronLayer& layer_above, bool include_bias )
{
    this->calc_layer_delta_values( layer_above );
    this->calc_layer_delta_weights( inputs );
    this->update_layer_weights( include_bias );
}