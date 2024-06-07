#pragma once
#include <iostream>

using namespace std;

template<class T>
void pair_print(pair<T, T> inp) {
	cout << inp.first << "x" << inp.second << endl;
}
