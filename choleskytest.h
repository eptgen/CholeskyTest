#ifndef CHOLESKYTEST_H
#define CHOLESKYTEST_H

#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

void cholesky(vector<vector<MatrixXd>> &A, int p, int b);
void choleskyOneThread(vector<vector<MatrixXd>> &A, int p, int b);

#endif