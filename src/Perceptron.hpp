#ifndef PERCEPTRON
#define PERCEPTRON

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
     * @brief Calculate and update the `delta` field.
     * delta = -1 * (desired_output - perceptron_output) * derivative_activity_function
     */
    void __calc_delta( double desired_output );

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

    /**
     * @brief Calculate and set the `delta_weights` field.
     *
     * @param inputs The vector of inputs to the perceptron.
     * @param desired_output The desired output from the perceptron.
     */
    void __calc_delta_weights( const std::vector< double >& inputs, double desired_output );

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
     * @brief Update the weights using the delta_weights field.
     *
     */
    void update_weights( const std::vector< double >& inputs, double desired_output );

};

#endif
