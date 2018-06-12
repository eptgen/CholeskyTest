#include <iostream>
#include <random>
#include <time.h>
#include <Eigen/Dense>
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

void cholesky(MatrixXd& A, int p, int b)
{
	// p * b = size of A
	for (int k = 0; k < p; k++)
	{
		// A.block(k * b, k * b, b, b) = A.block(k * b, k * b, b, b).llt().matrixL();
		Ref<MatrixXd> Akk = A.block(k * b, k * b, b, b);
		LLT<Ref<MatrixXd>> llt(Akk);
		for (int i = k + 1; i < p; i++)
		{
			A.block(i * b, k * b, b, b) = A.block(k * b, k * b, b, b).transpose().triangularView<Upper>().solve<OnTheRight>(A.block(i * b, k * b, b, b));
			// A.block(i * b, k * b, b, b) = A.block(i * b, k * b, b, b) * A.block(k * b, k * b, b, b).transpose().inverse();
		}
		for (int i = k + 1; i < p; i++)
		{
			for (int j = k + 1; j <= i; j++)
			{
				A.block(i * b, j * b, b, b) -= A.block(i * b, k * b, b, b) * A.block(j * b, k * b, b, b).transpose();
			}
		}
	}
	A = A.triangularView<Lower>();
}

int main()
{
	/*
	srand(time(NULL));
	
	int size = 16;
	MatrixXd testMatrix(size, size);
	
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			testMatrix(i, j) = rand() % 20 - 10;
		}
	}
	float smallNumber = 0.0005;
	
	testMatrix = testMatrix * testMatrix.transpose() + smallNumber * MatrixXd::Identity(size, size);
	
	MatrixXd copyTestMatrix(testMatrix);
	
	cout << "This is A: " << endl;
	cout << testMatrix << endl << endl;
	
	SelfAdjointEigenSolver<MatrixXd> eigensolver(testMatrix);
	cout << "Eigenvalues: " << endl;
	cout << eigensolver.eigenvalues() << endl << endl;
	
	MatrixXd answer = testMatrix.llt().matrixL();
	cout << "Real Answer: " << endl;
	cout << answer << endl << endl;
	
	cholesky(testMatrix, 4, 4); // 4 * 4 == 16 == size
	
	cout << "Test Answer: " << endl;
	cout << testMatrix << endl << endl;
	
	cout << "This should be A (real answer): " << endl;
	cout << answer * answer.transpose() << endl << endl;
	
	MatrixXd result = testMatrix * testMatrix.transpose();
	cout << "This should ALSO be A (test answer): " << endl;
	cout << testMatrix * testMatrix.transpose() << endl << endl;
	
	MatrixXd errors(size, size);
	
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			errors(i, j) = copyTestMatrix(i, j) - result(i, j);
		}
	}
	cout << "errors:" << endl;
	cout << errors << endl;
	*/
	
	test();
	
	return 0;
}



































