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
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

const int NUM_THREADS = 2;

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

struct gemm_args
{
	vector<vector<MatrixXd>>* A;
	int p;
	int k;
	int b;
	int startI; // inclusive
	int startJ; // inclusive
	int endI; // inclusive
	int endJ; // not inclusive
};

/*
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
*/

void manyGemms(gemm_args args)
{
	vector<vector<MatrixXd>> *A = (vector<vector<MatrixXd>>*) args.A;
	int k = args.k;
	int p = args.p;
	int b = args.b;
	int startI = args.startI;
	int startJ = args.startJ;
	int endI = args.endI;
	int endJ = args.endJ;
	for (int i = startI; i <= endI; i++)
	{
		int currStartJ = k + 1;
		int currEndJ = i + 1;
		if (i == startI)
		{
			currStartJ = startJ;
		}
		if (i == endI)
		{
			currEndJ = endJ;
		}
		for (int j = currStartJ; j < currEndJ; j++)
		{
			// cout << "(" << i << ", " << j << "): before: " << endl << (*A)[i][j] << endl;
			(*A)[i][j] -= (*A)[i][k] * (*A)[j][k].transpose();
			// cout << "after: " << endl << (*A)[i][j] << endl;
		}
	}
}

int triangle(int n)
{
	return (n * (n + 1)) / 2;
}

void cholesky(vector<vector<MatrixXd>>& A, int p, int b)
{
	// p * b = size of A
	double ttime = 0;
	thread* threads[NUM_THREADS];
	gemm_args argss[NUM_THREADS];
	int endI[NUM_THREADS];
	int endJ[NUM_THREADS];
	for (int k = 0; k < p; k++)
	{
		// cout << "======== k=" << k << " ========" << endl;
		// A.block(k * b, k * b, b, b) = A.block(k * b, k * b, b, b).llt().matrixL();
		LLT<Ref<MatrixXd>> llt(A[k][k]); // HMMMM
		for (int i = k + 1; i < p; i++)
		{
			A[i][k] = A[k][k].transpose().triangularView<Upper>().solve<OnTheRight>(A[i][k]);
			// A.block(i * b, k * b, b, b) = A.block(i * b, k * b, b, b) * A.block(k * b, k * b, b, b).transpose().inverse();
		}
		/*
		int size = p - k - 1;
		vector<pthread_t> threads;
		threads.reserve(size);
		vector<flags> completes;
		completes.reserve(size);
		vector<gemm_args> argss;
		argss.reserve(size);
		*/
		int numGemms = triangle(p - k - 1);
		int gemmsPerThread = numGemms / NUM_THREADS;
		int howManyOneExtra = numGemms % NUM_THREADS;
		int ind = 0;
		int current = 0;
		for (int i = k + 1; i < p; i++)
		{
			int tgpt = gemmsPerThread; // tentativeGemmsPerThread
			if (ind < howManyOneExtra)
			{
				tgpt++;
			}
			current += i - k;
			if (current >= tgpt)
			{
				while (current >= tgpt && ind < NUM_THREADS)
				{
					endI[ind] = i;
					endJ[ind] = (k + 1) + (i - k) - (current - tgpt);
					current -= tgpt;
					ind++;
					if (ind >= howManyOneExtra)
					{
						tgpt = gemmsPerThread;
					}
				}
			}
		}
		for (int i = 0; i < NUM_THREADS; i++)
		{
			gemm_args args;
			args.A = &A;
			args.k = k;
			args.b = b;
			args.p = p;
			if (i == 0)
			{
				args.startI = k + 1;
				args.startJ = k + 1;
			}
			else
			{
				args.startI = endI[i - 1];
				args.startJ = endJ[i - 1];
			}
			args.endI = endI[i];
			args.endJ = endJ[i];
			// cout << args.startI << "," << args.startJ << "-" << args.endI << "," << args.endJ << endl;
			argss[i] = args;
			thread* t = new thread(manyGemms, args);
			threads[i] = t;
			// pthread_create(&threads[i], NULL, manyGemms, (void*) &argss[i]);
			// pthread_join(threads[i], NULL);
		}
		/*
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
		*/
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
		
		timeval start;
		timeval end;
		gettimeofday(&start, 0);
		for (int i = 0; i < NUM_THREADS; i++)
		{
			threads[i]->join();
		}
		gettimeofday(&end, 0);
		ttime += timeSubtract(end, start);
		
		// cout << "=====================" << endl;
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
	// cout << "Total time thread calls took: " << ttime << " s" << endl;
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



































