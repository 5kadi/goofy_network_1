#include <iostream>
#include <vector>

using namespace std;

template<class T>
class Matrix {
public:
  Matrix(vector<T> data, pair<int, int> shape) : data(data), shape(shape) {
    reshape(shape);
  };
  Matrix(vector<vector<T>> matrix) : matrix(matrix) {
    shape = { matrix.size(), matrix[0].size() };
    data.clear();
    for (vector<T> y : matrix) {
      for (T x : y) {
        data.push_back(x);
      }
    }
  }
  void reshape(pair<int, int> new_shape) {
    if (new_shape.first * new_shape.second == data.size()) {
      matrix.clear();
      int offset = 0;
      for (int y{}; y < new_shape.first; y++) {
        vector<T> new_vector{};
        new_vector.clear();
        for (int x{}; x < new_shape.second; x++) {
          new_vector.push_back(data[x + offset]);
        }
        offset += new_shape.second;
        matrix.push_back(new_vector);
      }
      return;
    }
    throw runtime_error("Matrix shape and data size are different");
  }
  void flip() {
    vector<vector<T>> new_matrix = matrix;
    new_matrix.clear();
    for (int x{}; x < shape.second; x++) {
      vector<T> new_vector{};
      new_vector.clear();
      for (int y{}; y < shape.first; y++) {
        new_vector.push_back(matrix[y][x]);
      }
      new_matrix.push_back(new_vector);
    }
    matrix = new_matrix;
    shape = { shape.second, shape.first };
  }
  Matrix<T> operator * (Matrix<T> other) {
    if (shape.second == other.shape.first) {
      vector<vector<T>> new_matrix{};
      new_matrix.clear();
      other.flip();
      for (vector<T> this_y : matrix) {
        vector<T> new_vector{};
        new_vector.clear();
        for (vector<T> other_y : other.matrix) {
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
    throw runtime_error("Matrix_1 x should be equal to Matrix_2 y");
  }
  void print() {
    for (vector<T> y : matrix) {
      for (T x : y) {
        cout << x << " ";
      }
      cout << endl;
    }
  }
public:
  vector<T> data;
  pair<int, int> shape;
  vector<vector<T>> matrix{};
};

int main() {
  Matrix<int> m1{ {1, 2, 3, 4}, {2, 2}};
  Matrix<int> m2{ {{1, 2, 3}, {4, 5, 6}} };
  Matrix<int> m12 = m1 * m2;
  m12.print();
  m2.flip();
  Matrix<int> m122 = m12 * m2;
  m122.print();
}
