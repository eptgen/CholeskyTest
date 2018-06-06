#include <iostream>
#include <random>
#include <time.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

void cholesky(MatrixXd A, MatrixXd& L, int p, int b)
{
	// p * b = size of A
	for (int k = 0; k < p; k++)
	{
		L.block(k * b, k * b, b, b) = A.block(k * b, k * b, b, b).llt().matrixL();
		for (int i = k + 1; i < p; i++)
		{
			L.block(i * b, k * b, b, b) = A.block(i * b, k * b, b, b) * L.block(k * b, k * b, b, b).inverse().transpose();
		}
		for (int i = k + 1; i < p; i++)
		{
			for (int j = k + 1; j < p; j++)
			{
				A.block(i * b, j * b, b, b) = A.block(i * b, j * b, b, b) - L.block(i * b, k * b, b, b) * L.block(j * b, k * b, b, b).transpose();
			}
		}
	}
}

int main()
{
	srand(time(NULL));
	
	MatrixXd testMatrix(4, 4);
	while (true)
	{
		testMatrix(0, 0) = rand() % 20 - 10;
		testMatrix(0, 1) = rand() % 20 - 10;
		testMatrix(0, 2) = rand() % 20 - 10;
		testMatrix(0, 3) = rand() % 20 - 10;
		testMatrix(1, 0) = testMatrix(0, 1);
		testMatrix(1, 1) = rand() % 20 - 10;
		testMatrix(1, 2) = rand() % 20 - 10;
		testMatrix(1, 3) = rand() % 20 - 10;
		testMatrix(2, 0) = testMatrix(0, 2);
		testMatrix(2, 1) = testMatrix(1, 2);
		testMatrix(2, 2) = rand() % 20 - 10;
		testMatrix(2, 3) = rand() % 20 - 10;
		testMatrix(3, 0) = testMatrix(0, 3);
		testMatrix(3, 1) = testMatrix(1, 3);
		testMatrix(3, 2) = testMatrix(2, 3);
		testMatrix(3, 3) = rand() % 20 - 10;
		
		SelfAdjointEigenSolver<MatrixXd> eigensolver(testMatrix);
		VectorXd values = eigensolver.eigenvalues();
		
		bool shouldBreak = true;
		for (int i = 0; i < 4; i++)
		{
			if (values(i) < 0)
			{
				shouldBreak = false;
				break;
			}
		}
		if (shouldBreak)
		{
			break;
		}
	}
	
	cout << "This is A: " << endl;
	cout << testMatrix << endl << endl;
	
	SelfAdjointEigenSolver<MatrixXd> eigensolver(testMatrix);
	cout << "Eigenvalues: " << endl;
	cout << eigensolver.eigenvalues() << endl << endl;
	
	MatrixXd answer = testMatrix.llt().matrixL();
	cout << "Real Answer: " << endl;
	cout << answer << endl << endl;
	
	MatrixXd resMatrix(4, 4);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			resMatrix(i, j) = 0;
		}
	}
	cholesky(testMatrix, resMatrix, 2, 2);
	
	cout << "Test Answer: " << endl;
	cout << resMatrix << endl << endl;
	
	cout << "This should be A (real answer): " << endl;
	cout << answer * answer.transpose() << endl << endl;
	
	cout << "This should ALSO be A (test answer): " << endl;
	cout << resMatrix * resMatrix.transpose() << endl << endl;
	
	return 0;
}



































