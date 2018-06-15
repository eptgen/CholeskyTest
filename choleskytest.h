#ifndef CHOLESKYTEST_H
#define CHOLESKYTEST_H

#include <Eigen/Dense>

using namespace Eigen;

void cholesky(MatrixXd &A, int p, int b);
void choleskyOneThread(MatrixXd &a, int p, int b);

#endif