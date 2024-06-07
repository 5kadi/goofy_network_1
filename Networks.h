#pragma once
#ifndef NETWORKS_H
#define NETWORKS_H
#include "Layers.h"
class BaseNetwork {
public:
	BaseNetwork();
	BaseNetwork(LayerSequence seq);
	virtual Matrix<double> operator()(Matrix<double> input_values);
	virtual void backpropagate(Matrix<double> error);
	virtual void auto_create(); //wip
	LayerSequence seq;
};
class Network4x1 : public BaseNetwork {
public:
	Network4x1();
};
#endif 
