#include <iostream>
#include <random>
#include "Layers.h"
#include "utils.h"

using namespace std;

BaseLayer::BaseLayer(int inputs, int outputs) : inputs(inputs), outputs(outputs), links(inputs) {
	create_weights();
};
BaseLayer::~BaseLayer() {
};
void BaseLayer::create_weights() {
	vector<vector<double>> new_weights{};
	new_weights.clear();
	for (int y{}; y < outputs; y++) {
		vector<double> weights_vector{};
		weights_vector.clear();
		for (int x{}; x < inputs; x++) {
			random_device rd;
			mt19937 gen{ rd() };
			uniform_real_distribution<double> new_weight{ 0.0, 1.0/sqrt((double)links) };
			weights_vector.push_back(new_weight(gen));
		}
		new_weights.push_back(weights_vector);
	}
	weights = Matrix<double>{ new_weights };
}
void BaseLayer::backpropagate(Matrix<double> error) {
	if (error.shape.first * error.shape.second == outputs) { //т.к. в транспонированной матрице весов x = outputs
		vector<vector<double>> weights_ratio_vect{};
		weights_ratio_vect.clear();
		for (int k{}; k < outputs; k++) {
			vector<double> new_vector{};
			new_vector.clear();
			double new_sum = 0.0;
			for (int j{}; j < inputs; j++) {
				new_sum += weights[k][j];
			}
			for (int x{}; x < inputs; x++) {
				double weights_ratio_val = weights[k][x] / new_sum;
				new_vector.push_back(weights_ratio_val);
			}
			weights_ratio_vect.push_back(new_vector);
		}
		Matrix<double> weights_ratio{ weights_ratio_vect };
		weights_ratio.transpose();
		this->error = weights_ratio * error;
		return;
	}
}
Matrix<double> BaseLayer::operator() (Matrix<double> input_values) {
	if (input_values.shape.first * input_values.shape.second == inputs) {
		input_values.reshape({ inputs, 1 });
		signal = input_values;
		Matrix<double> output = weights * input_values;
		return output;
	}
	throw runtime_error("Matrix input_values size should be equal to Layer inputs");
}

InputLayer::InputLayer(int inputs, int outputs) : BaseLayer(inputs, outputs) {};

HiddenLayer::HiddenLayer(int inputs, int outputs) : BaseLayer(inputs, outputs) {};

OutputLayer::OutputLayer(int outputs) : BaseLayer(outputs, outputs) {};
void OutputLayer::backpropagate(Matrix<double> error) {
	this->error = error;
}
Matrix<double> OutputLayer::operator()(Matrix<double> input_values) {
	if (input_values.shape.first * input_values.shape.second == inputs) {
		input_values.reshape({ inputs, 1 });
		signal = input_values;
		return signal;
	}
	throw runtime_error("Matrix input_values size should be equal to Layer inputs");
}

LayerSequence::LayerSequence() : active_layers({}), num_active_layers(0) {
	active_layers.clear();
	output_layer = new OutputLayer{ 0 };
};
LayerSequence::LayerSequence(InputLayer* input_layer, vector<HiddenLayer*> hidden_layers, OutputLayer* output_layer) 
	: output_layer(output_layer) {
	active_layers = { input_layer };
	for (HiddenLayer* hidden_layer : hidden_layers) {
		active_layers.push_back(hidden_layer);
	}
	num_active_layers = active_layers.size();
	for (int idx = 1; idx < num_active_layers; idx++) {
		active_layers[idx]->links = active_layers[idx - 1]->outputs;
	}
	update();
};
void LayerSequence::update() {
	errors = {};
	errors.clear();
	for (int idx{}; idx < num_active_layers; idx++) {
		errors.push_back(active_layers[idx]->error);
	}
	errors.push_back(output_layer->error);
}
void LayerSequence::backpropagate(Matrix<double> error) {
	output_layer->backpropagate(error);
	error = output_layer->error;
	for (int idx = num_active_layers - 1; idx >= 0; idx--) {
		active_layers[idx]->backpropagate(error);
		error = active_layers[idx]->error;
	}
	update();
	return;
};
Matrix<double> LayerSequence::operator()(Matrix<double> input_values) {
	for (int idx{}; idx < num_active_layers; idx++) {
		input_values = active_layers[idx]->operator()(input_values);
	}
	input_values = output_layer->operator()(input_values);
	update();
	return input_values;
};
void LayerSequence::info() {
	for (int idx{}; idx < num_active_layers; idx++) {
		cout << typeid(active_layers[idx]).name() << endl;
	}
}
 