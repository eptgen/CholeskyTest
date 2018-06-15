#include <iostream>
#include <random>
#include <time.h>
#include <Eigen/Dense>
#include <vector>
#include <pthread.h>
#include <thread>
#include <chrono>
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

void choleskyOneThread(MatrixXd& A, int p, int b)
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

struct flags
{
	bool complete;
};

struct gemm_args
{
	MatrixXd* A;
	int i;
	int p;
	int k;
	int b;
	flags* f;
};

void *manyGemms(void *arguments)
{
	gemm_args *args = (gemm_args*) arguments;
	MatrixXd *A = (MatrixXd*) args->A;
	int i = args->i;
	int p = args->p;
	int k = args->k;
	int b = args->b;
	for (int j = k + 1; j <= i; j++)
	{
		// cout << "A" << i << j << " -= A" << i << k << " * A" << j << k << "^T" << endl;
		A->block(i * b, j * b, b, b) -= A->block(i * b, k * b, b, b) * A->block(j * b, k * b, b, b).transpose();
		// cout << "i=" << i << ", j=" << j << ": setting " << args->f << " from " << args->f->complete << " to 1" << endl;
	}
	args->f->complete = true;
	// cout << "done setting complete to " << args->f->complete << endl;
	pthread_exit(NULL);
}

int triangle(int n)
{
	return (n * (n + 1)) / 2;
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
		int size = p - k - 1;
		vector<pthread_t> threads;
		threads.reserve(size);
		vector<flags> completes;
		completes.reserve(size);
		vector<gemm_args> argss;
		argss.reserve(size);
		for (int i = p - 1; i > k; i--)
		{
			flags f;
			f.complete = false;
			completes.push_back(f);
			pthread_t thread;
			threads.push_back(thread);
			gemm_args args;
			args.A = &A;
			args.i = i;
			args.k = k;
			args.b = b;
			args.f = &completes[completes.size() - 1];
			args.p = p;
			argss.push_back(args);
			pthread_create(&threads[threads.size() - 1], NULL, manyGemms, (void*) &argss[argss.size() - 1]);
		}
		/*
		cout << "threads: " << endl;
		for (int i = 0; i < size; i++)
		{
			cout << &threads[i] << endl;
		}
		cout << "completes" << endl;
		for (int i = 0; i < size; i++)
		{
			cout << &completes[i] << endl;
		}
		*/
		// cout << "out of the loop " << k << endl;
		int i = 0; // to ensure that it's not checking the same thing twice
		// cout << "====================" << endl;
		// cout << completes.size() << endl;
		// cout << size << endl;
		// cout << "====================" << endl;
		while (true && size > 0)
		{
			/*
			cout << completes.size() << ":" << endl;
			for (int i = 0; i < completes.size(); i++)
			{
				if (completes[i].complete)
				{
					cout << "true ";
				}
				else
				{
					cout << "false ";
				}
			}
			cout << endl;
			*/
			/*
			bool shouldBreak = true;
			for (int i = start; i < completes.size(); i++)
			{
				if (!completes[i].complete)
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
			*/
			// cout << i << endl;
			if (completes[i].complete)
			{
				i++;
				if (i == size)
				{
					break;
				}
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



































