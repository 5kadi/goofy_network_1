#pragma once
#ifndef OPTIM_H
#define OPTIM_H
#include "Networks.h"
class Optim {
public:
	Optim(BaseNetwork* network, double lr);
	void step();
	Matrix<double> calculate_derivative(BaseLayer layer, Matrix<double> next_layer_error);
	double sigmoid(double x);
	BaseNetwork* network;
	LayerSequence* seq; 
	double lr;
};
#endif 

