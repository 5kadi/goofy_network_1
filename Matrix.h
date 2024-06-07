#pragma once
#ifndef MATRIX_H
#define MATRIX_H
#include <iostream>
#include <vector>
#include "utils.h"
template<class T>
class Matrix {
public:
	Matrix() {
		data = { 0 };
		shape = { 1, 1 };
		matrix = { {0} };
	};
	Matrix(std::vector<T> data, std::pair<int, int> shape) {
		reshape(shape);
	};
	Matrix(std::vector<std::vector<T>> matrix) : matrix(matrix) {
		shape = { matrix.size(), matrix[0].size() };
		data.clear();
		for (std::vector<T> y : matrix) {
			for (T x : y) {
				data.push_back(x);
			}
		}
	};
	void reshape(std::pair<int, int> new_shape) {
		if (new_shape.first * new_shape.second == data.size()) {
			matrix.clear();
			int offset = 0;
			for (int y{}; y < new_shape.first; y++) {
				std::vector<T> new_vector{};
				new_vector.clear();
				for (int x{}; x < new_shape.second; x++) {
					new_vector.push_back(data[x + offset]);
				}
				offset += new_shape.second;
				matrix.push_back(new_vector);
			}
			return;
		}
		throw std::runtime_error("Matrix shape and data size are different");
	}
	void transpose() {
		std::vector<std::vector<T>> new_matrix = matrix;
		new_matrix.clear();
		for (int x{}; x < shape.second; x++) {
			std::vector<T> new_vector{};
			new_vector.clear();
			for (int y{}; y < shape.first; y++) {
				new_vector.push_back(matrix[y][x]);
			}
			new_matrix.push_back(new_vector);
		}
		matrix = new_matrix;
		shape = { shape.second, shape.first };
	}
	void print() {
		for (std::vector<T> y : matrix) {
			for (T x : y) {
				std::cout << x << " ";
			}
			std::cout << std::endl;
		}
	}
	Matrix<T> operator * (Matrix<T> other) {
		if (shape.second == other.shape.first) {
			std::vector<std::vector<T>> new_matrix{};
			new_matrix.clear();
			other.transpose();
			for (std::vector<T> this_y : matrix) {
				std::vector<T> new_vector{};
				new_vector.clear();
				for (std::vector<T> other_y : other.matrix) {
					T new_element = 0;
					for (int x{}; x < other_y.size(); x++) {
						new_element += this_y[x] * other_y[x];
					}
					new_vector.push_back(new_element);
				}
				new_matrix.push_back(new_vector);
			}
			Matrix<T> ret = Matrix(new_matrix);
			return ret;
		}
		pair_print(shape);
		pair_print(other.shape);
		throw std::runtime_error("Matrix_1 x should be equal to Matrix_2 y");
	};
	Matrix<T> operator - (Matrix<T> other) {
		if (shape == other.shape) {
			std::vector<std::vector<T>> new_matrix{};
			new_matrix.clear();
			for (int y{}; y < shape.first; y++) {
				std::vector<T> new_vector{};
				new_vector.clear();
				T new_element = 0;
				for (int x{}; x < shape.second; x++) {
					new_element = this[0][y][x] - other[y][x];
					new_vector.push_back(new_element);
				}
				new_matrix.push_back(new_vector);
			}
			return Matrix{ new_matrix };
		}
		pair_print(shape);
		pair_print(other.shape);
		cout << "error" << endl;
	}
	auto& operator[](int index) {
		return matrix[index];
	}

	std::vector<T> data;
	std::pair<int, int> shape;
	std::vector<std::vector<T>> matrix{};
};
#endif