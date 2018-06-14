#include <iostream>
#include <random>
#include <time.h>
#include <Eigen/Dense>
#include <vector>
#include <pthread.h>
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

struct gemm_args
{
	MatrixXd* A;
	int i;
	int j;
	int k;
	int b;
	bool* complete;
};

void *gemm(void *arguments)
{
	gemm_args *args = (gemm_args*) arguments;
	MatrixXd *A = (MatrixXd*) args->A;
	int i = (long) args->i;
	int j = (long) args->j;
	int k = (long) args->k;
	int b = (long) args->b;
	bool *complete = args->complete;
	// cout << "A" << i << j << " -= A" << i << k << " * A" << j << k << "^T" << endl;
	A->block(i * b, j * b, b, b) -= A->block(i * b, k * b, b, b) * A->block(j * b, k * b, b, b).transpose();
	// cout << "setting complete to true" << endl;
	complete = new bool(true);
	pthread_exit(NULL);
}

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
		vector<pthread_t> threads;
		// threads.reserve(p * p);
		vector<bool*> completes;
		// completes.reserve(p * p);
		for (int i = k + 1; i < p; i++)
		{
			for (int j = k + 1; j <= i; j++)
			{
				// cout << "i=" << i << ", j=" << j << endl;
				bool *f = new bool(false);
				// cout << "bool *f = new bool(false);" << endl;
				completes.push_back(f);
				// cout << "completes.push_back(f);" << endl;
				pthread_t thread;
				// cout << "pthread_t thread;" << endl;
				// cout << "threads.size() == " << threads.size() << endl;
				threads.push_back(thread);
				// cout << "threads.push_back(thread);" << endl;
				gemm_args args;
				// cout << "gemm_args args;" << endl;
				args.A = &A;
				// cout << "args.A = &A;" << endl;
				args.i = i;
				args.j = j;
				args.k = k;
				args.b = b;
				args.complete = completes[i];
				// cout << "creating thread" << endl;
				pthread_create(&threads[i], NULL, gemm, (void*) &args);
				// cout << "I am done creating the thread " << i << " " << j << " " << k << " " << p << endl;
			}
		}
		// cout << "out of the loop " << k << endl;
		int start = 0; // to ensure that it's not checking the same thing twice
		while (true)
		{
			bool shouldBreak = true;
			/*
			for (int i = k + 1; i < p; i++)
			{
				for (int j = k + 1; j <= i; j++)
				{
					if (!completes[i * p + j])
					{
						start = i;
						shouldBreak = false;
						break;
					}
				}
				if (!shouldBreak)
				{
					break;
				}
			}
			*/
			for (int i = start; i < threads.size(); i++)
			{
				if (!(*completes[i]))
				{
					start = i;
					shouldBreak = false;
					break;
				}
			}
			if (shouldBreak)
			{
				break;
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
	
	cholesky(testMatrix, 4, 4); // 2 * 2 == 4 == size
	
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



































