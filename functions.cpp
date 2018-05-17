#include <cmath>
#include "functions.hpp"


double Sigmoid::operator() (double x)
{
    return 1.0 / (1.0 + exp(-x));
}


double Sigmoid::derivative (double x)
{
    return (*this)(x) * (1.0 - (*this)(x));
}
