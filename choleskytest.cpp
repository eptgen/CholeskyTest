#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/time.h>
#include <unistd.h>
#include <map>
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

const int NUM_THREADS = 2;

const int LLT = 0;
const int GEMM = 1;
const int TRSM = 2;
const int FINISH = 3;

void choleskyOneThread(vector<vector<MatrixXd>>& A, int p, int b)
{
	// p * b = size of A
	for (int k = 0; k < p; k++)
	{
		// A.block(k * b, k * b, b, b) = A.block(k * b, k * b, b, b).llt().matrixL();
		Ref<MatrixXd> Akk = A[k][k];
		LLT<Ref<MatrixXd>> llt(Akk);
		for (int i = k + 1; i < p; i++)
		{
			A[i][k] = A[k][k].transpose().triangularView<Upper>().solve<OnTheRight>(A[i][k]);
			// A.block(i * b, k * b, b, b) = A.block(i * b, k * b, b, b) * A.block(k * b, k * b, b, b).transpose().inverse();
		}
		for (int i = k + 1; i < p; i++)
		{
			for (int j = k + 1; j <= i; j++)
			{
				A[i][j] -= A[i][k] * A[j][k].transpose();
			}
		}
	}
	for (int i = 0; i < p; i++)
	{
		A[i][i] = A[i][i].triangularView<Lower>();
	}
	for (int i = 0; i < p; i++)
	{
		for (int j = i + 1; j < p; j++)
		{
			A[i][j] = MatrixXd::Zero(b, b);
		}
	}
}

struct Op
{
	int id;
	int i; // only when id != 3
	int j; // only when id != 3
	vector<Op> dependencies;
	bool operator<(const Op& other)
	{
		return id < other.id || (id == other.id && i < other.i) || (id == other.id && i == other.i && j < other.j);
	}
}

void goThroughTasks(vector<Op>* tasks)
{
	
}

void cholesky(vector<vector<MatrixXd>>& A, int p, int b)
{
	set<long long> completed
	vector<Op> operationsToDo;
	thread* threads[NUM_THREADS];
	for (int i = 0; i < p; i++)
	{
		
	}
	for (int i = 0; i < NUM_THREADS; i++)
	{
		thread* t = new thread(goThroughTasks, &operationsToDo);
		threads[i] = t;
	}
}

int main()
{
	// cout << "PROGRAM: START" << endl;
	
	/*
	srand(time(NULL));
	
	int p = 4;
	int b = 2;
	// p * b == size
	MatrixXd conMatrix(p * b, p * b); // convenience matrix
	vector<vector<MatrixXd>> testMatrix;
	
	// cout << "creating convenience matrix" << endl;
	for (int i = 0; i < p * b; i++)
	{
		for (int j = 0; j < p * b; j++)
		{
			conMatrix(i, j) = rand() % 20 - 10;
		}
	}
	float smallNumber = 0.0005;
	
	conMatrix = conMatrix * conMatrix.transpose() + smallNumber * MatrixXd::Identity(p * b, p * b);
	
	// cout << "creating testMatrix" << endl;
	for (int i = 0; i < p; i++)
	{
		vector<MatrixXd> v;
		testMatrix.push_back(v);
		for (int j = 0; j < p; j++)
		{
			MatrixXd m(b, b);
			testMatrix[i].push_back(m);
			for (int k = 0; k < b; k++)
			{
				for (int l = 0; l < b; l++)
				{
					testMatrix[i][j](k, l) = conMatrix(i * b + k, j * b + l);
				}
			}
		}
	}
	
	cout << "This is A: " << endl;
	cout << conMatrix << endl << endl;
	
	SelfAdjointEigenSolver<MatrixXd> eigensolver(conMatrix);
	cout << "Eigenvalues: " << endl;
	cout << eigensolver.eigenvalues() << endl << endl;
	
	MatrixXd answer = conMatrix.llt().matrixL();
	cout << "Real Answer: " << endl;
	cout << answer << endl << endl;
	
	cholesky(testMatrix, p, b);
	
	MatrixXd otherConMatrix(p * b, p * b);
	for (int i = 0; i < p; i++)
	{
		for (int j = 0; j < p; j++)
		{
			for (int k = 0; k < b; k++)
			{
				for (int l = 0; l < b; l++)
				{
					otherConMatrix(i * b + k, j * b + l) = testMatrix[i][j](k, l);
				}
			}
		}
	}
	
	cout << "Test Answer: " << endl;
	cout << otherConMatrix << endl << endl;
	
	cout << "This should be A (real answer): " << endl;
	cout << answer * answer.transpose() << endl << endl;
	
	
	MatrixXd result = otherConMatrix * otherConMatrix.transpose();
	cout << "This should ALSO be A (test answer): " << endl;
	cout << result << endl << endl;
	
	MatrixXd errors(p * b, p * b);
	
	for (int i = 0; i < p * b; i++)
	{
		for (int j = 0; j < p * b; j++)
		{
			errors(i, j) = conMatrix(i, j) - result(i, j);
		}
	}
	cout << "errors:" << endl;
	cout << errors << endl;
	*/
	
	test();
	
	return 0;
}



































