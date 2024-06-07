#include "Optim.h"

Optim::Optim(BaseNetwork* network, double lr) : network(network), seq(&network->seq), lr(lr) {};
void Optim::step() {
	for (int l_idx{}; l_idx < seq->num_active_layers; l_idx++) {
		BaseLayer* current_layer = seq->active_layers[l_idx];
		Matrix<double> next_layer_error = seq->errors[l_idx + 1];
		Matrix<double> new_weights = current_layer->weights - calculate_derivative(*current_layer, next_layer_error);
		current_layer->weights = new_weights;

	}
};
Matrix<double> Optim::calculate_derivative(BaseLayer layer, Matrix<double> next_layer_error) {
	vector<vector<double>> derivative_matrix{};
	derivative_matrix.clear();
	Matrix<double> weights = layer.weights;
	Matrix<double> signal = layer.signal;
	for (int k{}; k < layer.outputs; k++) {
		vector<double> derivative_vector{};
		derivative_vector.clear();
		double new_sum = 0.0;
		for (int j{}; j < layer.inputs; j++) {
			new_sum += weights[k][j] * signal[j][0];
		}
		for (int j{}; j < layer.inputs; j++) {
			double new_derivative = lr * -1 * next_layer_error[k][0] * sigmoid(new_sum) * (1 - sigmoid(new_sum)) * signal[j][0];
			derivative_vector.push_back(new_derivative);
		}
		derivative_matrix.push_back(derivative_vector);
	}
	return Matrix<double>{derivative_matrix};
};
double Optim::sigmoid(double x) {
	return 1.0 / (1 + exp(-1 * x));
};