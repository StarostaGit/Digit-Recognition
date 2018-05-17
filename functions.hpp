#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP


/** An interface of a function object with its partial value      **/
/** and derivative. Every mathematical function used in Network   **/
/** class should implement this in order for it to work properly  **/

class Function
{
    public:

        // Returns a value of a function in a given point
        virtual double operator() (double x) = 0;

        // Returns a partial derivative in a given point
        virtual double derivative (double x) = 0;
};


/** A sigmoid activation function implementing the Function interface **/

class Sigmoid : public virtual Function
{

    public:

        virtual double operator() (double x) override;
        virtual double derivative (double x) override;

};


#endif // FUNCTIONS_HPP
