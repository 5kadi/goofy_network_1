#pragma once
#ifndef LOSS_H
#define LOSS_H
#include "Matrix.h"
#include "Networks.h"
class Loss {
public:
	Loss();
	Loss(BaseNetwork* network);
	void backpropagate();
	Matrix<double> operator()(Matrix<double> output, Matrix<double> reference);
	BaseNetwork* network;
	Matrix<double> error;
};
#endif
