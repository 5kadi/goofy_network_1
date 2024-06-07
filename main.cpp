#include <iostream>
#include "Layers.h"
#include "Loss.h"
#include "Optim.h"
#include "Networks.h"

using namespace std;

typedef struct test {
	int a;
} TEST;


int main() {
	Network4x1 nn{};
	Loss loss{ &nn };
	Optim optim{ &nn, 0.2638 };

	Matrix<double> vals{ {{1.0}, {2.0}, {3.0}, {4.0}} };
	Matrix<double> ref{ {{10.0}} };

	for (int epoch{}; epoch < 30; epoch++) {
		auto out = nn(vals);
		loss(out, ref);
		loss.backpropagate();
		optim.step();
		loss.error.print();
	}


	nn({ {{2.0}, {2.0}, {2.0}, {2.0}} }).print();
}