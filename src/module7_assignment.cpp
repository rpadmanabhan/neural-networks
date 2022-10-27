#include "PerceptronLayer.hpp"

#include <iostream>
#include <iomanip>
#include <iterator>

// for printing a vector
template<typename T>
std::ostream & operator<<(std::ostream & os, std::vector<T> vec)
{
    os<<"{ ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
    os<<"}";
    return os;
}

// The Feed Forward BackPropogation Algorithm
void ffbp(
    std::vector< PerceptronLayer >& neural_network,
    std::vector< double > input,
    std::vector< double > output,
    bool include_bias )
{
    // feed forward
    int idx = 0;

    for ( auto & layer : neural_network )
    {
        if ( idx == 0 )
            layer.calc_layer_output( input );
        else
            layer.calc_layer_output( neural_network[ idx - 1].output );

        ++idx;
    }

    // back propogation
    int last_idx = neural_network.size() - 1;
    for ( idx = last_idx; idx >= 0; --idx )
    {

        if ( idx == 0 )
            neural_network[ idx ].update( input, neural_network[ idx + 1 ], include_bias );
        else if ( idx == last_idx )
            neural_network[ idx ].update( neural_network[ idx - 1 ].output , output, include_bias );
        else
            neural_network[ idx ].update( neural_network[ idx - 1 ].output, neural_network[ idx + 1 ], include_bias );

        size_t j = 0;
    }
}

int main()
{
    std::string linebreak( "\n-----------------------------------------------------------------------------------------\n" );
    // note: this only sets precision for printing to screen.
    std::cout << std::setprecision(4) << std::fixed;

    std::cout << "Module7 Assignment\n";

    // Initial test.
    std::cout << linebreak;
    std::cout << "Initial Test without Bias: \n";
    PerceptronLayer layer1 = { 2, { { 0.3, 0.3 }, { 0.3, 0.3 } }, { 0, 0 }, 1.0, false };
    PerceptronLayer layer2 = { 1, { { 0.8, 0.8 } }, { 0 }, 1.0, true  };

    std::vector< PerceptronLayer > neural_network;
    neural_network.push_back( { 2, { { 0.3, 0.3 }, { 0.3, 0.3 } }, { 0, 0 }, 1.0, false } );
    neural_network.push_back( { 1, { { 0.8, 0.8 } }, { 0 }, 1.0, true  } );

    ffbp( neural_network, { 1.0, 2.0 }, { 0.7 }, false  );
    std::cout << "Output: " << neural_network[ 0 ].output << "\n";
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( {0.7} ) << "\n";

    ffbp( neural_network, { 1.0, 2.0 }, { 0.7 }, false  );
    std::cout << "Output: " << neural_network[ 0 ].output << "\n";
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( {0.7} );

    std::cout << linebreak;


    // Questions.

    neural_network.clear();
    neural_network.push_back( { 2, { { 0.3, 0.3 }, { 0.3, 0.3 } }, { 0, 0 }, 1.0, false } );
    neural_network.push_back( { 1, { { 0.8, 0.8 } }, { 0 }, 1.0, true  } );

    // Method 1.
    std::cout << linebreak;
    std::cout << "Training Method 1: \n";
    for ( int i = 0; i < 15; ++i )
    {
        ffbp( neural_network, { 1.0, 1.0 }, { 0.9 }, true  );
        ffbp( neural_network, { -1.0, -1.0 }, { 0.05 }, true  );
    }

    neural_network[ 0 ].calc_layer_output( { 1.0, 1.0 } );
    neural_network[ 1 ].calc_layer_output( neural_network[ 0 ].output );
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( { 0.9 } ) << "\n";

    neural_network[ 0 ].calc_layer_output( { -1.0, -1.0 } );
    neural_network[ 1 ].calc_layer_output( neural_network[ 0 ].output );
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( { 0.05 } );

    std::cout << linebreak;


    // Method 2.
    std::cout << linebreak;
    std::cout << "Training Method 2: \n";

    neural_network.clear();
    neural_network.push_back( { 2, { { 0.3, 0.3 }, { 0.3, 0.3 } }, { 0, 0 }, 1.0, false } );
    neural_network.push_back( { 1, { { 0.8, 0.8 } }, { 0 }, 1.0, true  } );
    for ( int i = 0; i < 15; ++i )
    {
        ffbp( neural_network, { 1.0, 1.0 }, { 0.9 }, true  );
    }
    for ( int i = 0; i < 15; ++i )
    {
        ffbp( neural_network, { -1.0, -1.0 }, { 0.05 }, true  );
    }

    neural_network[ 0 ].calc_layer_output( { 1.0, 1.0 } );
    neural_network[ 1 ].calc_layer_output( neural_network[ 0 ].output );
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( { 0.9 } ) << "\n";

    neural_network[ 0 ].calc_layer_output( { -1.0, -1.0 } );
    neural_network[ 1 ].calc_layer_output( neural_network[ 0 ].output );
    std::cout << "Output: " << neural_network[ 1 ].output << "\n";
    std::cout << "Error: " << neural_network[ 1 ].get_layer_bigE( { 0.05 } );

    std::cout << linebreak;


    return 0;
}