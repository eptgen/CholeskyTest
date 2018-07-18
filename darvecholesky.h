#ifndef DARVECHOLESKY_H
#define DARVECHOLESKY_H

#include <Eigen/Dense>

using namespace Eigen;

void cholesky(MatrixXd& A, int p, int b, int n_thread);

#endif