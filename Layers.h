#pragma once
#ifndef LAYERS_H
#define LAYERS_H
#include "Matrix.h"
#include <vector>
class BaseLayer {
protected:
	BaseLayer(int inputs, int outputs);
public:
	~BaseLayer();
	virtual void create_weights();
	virtual void backpropagate(Matrix<double> error);
	virtual Matrix<double> operator()(Matrix<double> input_values);


	Matrix<double> weights;
	Matrix<double> signal;
	Matrix<double> error;
	int links;
	int inputs;
	int outputs;
};
class InputLayer : public BaseLayer {
public:
	InputLayer(int inputs, int outputs);
};
class HiddenLayer : public BaseLayer {
public:
	HiddenLayer(int inputs, int outputs);
};
class OutputLayer : public BaseLayer {
public:
	OutputLayer(int outputs);
	void backpropagate(Matrix<double> error) override;
	Matrix<double> operator()(Matrix<double> input_values) override;
};

class LayerSequence {
public:
	LayerSequence();
	LayerSequence(InputLayer* input_layer, std::vector<HiddenLayer*> hidden_layers, OutputLayer* output_layer);
	void update();
	void backpropagate(Matrix<double> error);
	Matrix<double> operator()(Matrix<double> input_values);
	void info();

	int num_active_layers;
	OutputLayer* output_layer;
	std::vector<BaseLayer*> active_layers;
	std::vector<Matrix<double>> errors;
	std::vector<Matrix<double>> signals;
};
#endif
