#pragma once

#include "Perceptron.hpp"


class PerceptronLayer
{
public:
    // core fields
    std::vector< Perceptron > perceptrons; // all the perceptrons in this layer.
    std::vector< std::vector< double > > weights;  // weight vectors for each of the preceptrons.
    std::vector< double > biases;  // bias value for each of the preceptrons.
    std::vector< std::vector< double > > previous_weights;  // weight vectors for each of the preceptrons, previous iteration.
    std::vector< double > previous_biases;  // bias value for each of the preceptrons, previous iteration.

    bool is_output_layer; // whether this layer is an output layer or not (otherwise a hidden layer).
    std::vector< double > error; // only meaningful when is_output_layer = true.
    std::vector< double > output; // output of each perceptron in this layer.

    // store these for convenience.
    double eta;
    std::string activation_function;

    /**
     * @brief Construct a new Perceptron Layer object.
     * Will initialize multiple Perceptron objects.
     * Note: The number of inputs for each Perceptron is inferred by the dimension of the initial weight vector for each perceptron.
     * @param num_perceptrons 
     * @param initial_weights 
     * @param initial_biases 
     * @param eta 
     * @param is_output_layer 
     * @param activation_funtion 
     */
    PerceptronLayer(
        size_t num_perceptrons,
        const std::vector< std::vector< double > >& initial_weights,
        const std::vector< double >& initial_biases,
        double eta,
        bool is_output_layer,
        const std::string& activation_funtion = "Sigmoid"
    );


    /**
     * @brief Compute and set the error field for this layer - only meaningful if this is an output layer.
     * 
     * @param desired_output 
     */
    void calc_error( const std::vector< double >& desired_output );

    /**
     * @brief Compute and set the output field for this layer.
     * 
     * @param input 
     */
    void calc_layer_output( const std::vector< double >& inputs );

    /**
     * @brief Convenience function for getting the number of perceptrons in this layer.
     * 
     * @return size_t 
     */
    size_t get_layer_length();

    /**
     * @brief Compute delta values for all the perceptrons in an output layer.
     * 
     */
    void calc_layer_delta_values( const std::vector< double >& desired_output );

    /**
     * @brief Compute delta values for all the perceptrons in a hidden layer.
     * 
     * @param previous_layer 
     */
    void calc_layer_delta_values( const PerceptronLayer& layer_above );

    /**
     * @brief Compute delta weights for all the perceptrons in this layer.
     * 
     * @param input 
     */
    void calc_layer_delta_weights( const std::vector< double >& inputs );

    /**
     * @brief Update weights for all the perceptrons in this layer.
     * 
     */
    void update_layer_weights( bool include_bias = false );
};