#include "Networks.h"

BaseNetwork::BaseNetwork() {
	InputLayer i{ 1, 1 };
	HiddenLayer h1{ 1, 1 };
	OutputLayer o{ 1 };
	seq = { &i, {&h1}, &o };
};
BaseNetwork::BaseNetwork(LayerSequence seq) : seq(seq) {};
Matrix<double> BaseNetwork::operator()(Matrix<double> input_values) {
	return seq(input_values);
};
void BaseNetwork::backpropagate(Matrix<double> error) {
	seq.backpropagate(error);
}
void BaseNetwork::auto_create() {
	return; //wip
}

Network4x1::Network4x1() : BaseNetwork({ new InputLayer{ 4, 4 },
	{new HiddenLayer{ 4, 2 }, new HiddenLayer{ 2, 1 }},
	new OutputLayer{ 1 } }) {
}

