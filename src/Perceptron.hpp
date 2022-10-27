#pragma once

#include <string>
#include <vector>
#include <memory>

#include <math.h>

struct ActivationFunction
{
    virtual double value( double x ) const
    {
        return 1;
    }

    virtual double derivative( double x ) const
    {
        return 0;
    }
};


class Perceptron
{
private:
    std::unique_ptr< ActivationFunction > __activation_function_ptr;

    /**
     * @brief Calculate and update the `activity` field.
     *
     * @param inputs The vector of inputs to the perceptron.
     */
    void __calc_activity( const std::vector< double >& inputs );

    /**
     * @brief Calculate and update the `activation` field.
     *
    */
    void __calc_activation();

public:
    std::vector< double > weights;
    std::vector< double > delta_weights; // delta value for the weights

    double delta; // common delta value term for all the weights and bias (same as delta_bias)
    double bias;
    double activity;
    double activation;
    double eta;

    /**
     * @brief Construct a new Perceptron object.
     *
     * @param weights Vector of initial weights to set.
     * @param bias Bias value for the perceptron.
     * @param activation_funtion The activation function for the perceptron. [ default: "sigmoid" ]
     */
    Perceptron(
        const std::vector< double >& weights,
        double bias,
        double eta,
        const std::string& activation_funtion = "Sigmoid"
    );

    /**
     * @brief Return the output from this perceptron.
     * 
     * @param inputs The vector of inputs to the perceptron
     * @return the computed activation function value.
     */
    double get_output( const std::vector< double >& inputs );

    /**
     * @brief Compute the error w.r.t to a desired output.
     * 
     * @param desired_output The desired output from the perceptron
     * @return The error term
     */
    double get_error( double desired_output ) const;

    /**
     * @brief Compute the bigE term w.r.t to a desired output. i.e. bigE = 1/2(error^2)
     *
     * @param desired_output The desired output from the perceptron
     * @return The bigE term
     *
    */
    double get_bigE( double desired_output ) const;


    /**
     * @brief Calculate and update the `delta` field.
     * delta = -1 * (desired_output - perceptron_output) * derivative_activity_function
     * @param desired_output The desired output from the perceptron.
     */
    void calc_delta( double desired_output );

    /**
     * @brief Calculate and set the `delta_weights` field.
     *
     * @param inputs The vector of inputs to the perceptron.
     * @param desired_output The desired output from the perceptron.
     */
    void calc_delta_weights( const std::vector< double >& inputs );

    /**
     * @brief Update the weights using the delta_weights field.
     *
     * @param include_bias Whether to include the bias term in the update.
     */
    void update_weights( bool include_bias = true );

};