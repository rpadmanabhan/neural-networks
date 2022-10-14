#include "PerceptronLayer.hpp"

#include <iostream>
#include <iomanip>


int main()
{
    // note: this only sets precision for printing to screen.
    std::cout << std::setprecision(4) << std::fixed;

    auto hidden_layer = PerceptronLayer( 2, { { 0.8, 0.1 }, { 0.5, 0.2 } }, { 0, 0 }, 0.1, false );

    // Q1, Q2.
    hidden_layer.calc_layer_output( { 1.0, 3.0 } );
    size_t i = 0;
    for ( auto e : hidden_layer.output )
    {
        std::cout << "Node" << i + 1 << " Initial activation function value: " << e << "\n";
        ++i;
    }

    // Q3.
    auto output_layer = PerceptronLayer( 1, { { 0.2, 0.7 } }, { 0 }, 0.1, true );
    output_layer.calc_layer_output( hidden_layer.output );
    for ( auto e : output_layer.output )
    {
        std::cout << "Node3 initial activation function value: " << e << "\n";
    }

    // Q4, Q5.
    output_layer.calc_layer_delta_values( { 0.95 } );
    output_layer.calc_layer_delta_weights( hidden_layer.output );
    output_layer.update_layer_weights( false );

    for ( auto & weights : output_layer.weights )
    {
        size_t i = 0;
        for ( auto weight : weights )
        {
            std::cout << "Updated weight from hidden layer node " << i + 1  << " to node3 " << weight << "\n";
            ++i;
        }
    }


    // Q6.
    hidden_layer.calc_layer_delta_values( output_layer );
    hidden_layer.calc_layer_delta_weights( { 1.0, 3.0 } );
    hidden_layer.update_layer_weights( false );

    for ( auto & weights : hidden_layer.weights )
    {
        size_t i = 0;
        for ( auto weight : weights )
            std::cout << "Updated weights in hidden layer for node1 from input node" << i + 1 << ": " << weight  << "\n";
    }


    return 0;
}