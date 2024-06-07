#include "Loss.h"

Loss::Loss() {
	network = NULL;
};
Loss::Loss(BaseNetwork* network) : network(network) {};
void Loss::backpropagate() {
	if (network != NULL) {
		network->seq.backpropagate(error);
	}
};
Matrix<double> Loss::operator()(Matrix<double> output, Matrix<double> reference) {
	if (output.shape == reference.shape) {
		error = reference - output;
		return error;
	}
};