#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <Eigen/Dense>
#include <vector>
#include <thread>
#include <queue>
#include <chrono>
#include <sys/time.h>
#include <unistd.h>
#include <tuple>
#include <map>
#include <mutex>
#include <set>
#include "unittests.h"
#include "choleskytest.h"

using namespace std;
using namespace Eigen;

const int NUM_THREADS = 2;

const int POTF = 0;
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
	int i;
	int j;
	int k;
	int cReq;
	vector<tuple<int, int, int, int>> connect;
	bool finConnect;
};

struct OpArgs
{
	queue<Op>* waitingOps;
	mutex* waitingMutex;
	queue<Op>* myOps;
	mutex* myMutex;
	vector<vector<MatrixXd>>* A;
	int p;
	map<tuple<int, int, int, int>, int>* ops;
	set<tuple<int, int, int, int>>* alreadyPut;
	mutex* countMutex;
	mutex* setMutex;
};

void getPOTF(Op& o, int p)
{
	// o.id, o.i, o.j, and o.k should be filled out already
	
	if (o.k == 0)
	{
		o.cReq = 0;
	}
	else
	{
		o.cReq = 1;
	}
	
	for (int i = o.k + 1; i < p; i++)
	{
		tuple<int, int, int, int> t = make_tuple(TRSM, i, 0, o.k);
		o.connect.push_back(t);
	}
	o.finConnect = o.k == p - 1;
}

void getTRSM(Op& o, int p)
{
	// o.id, o.i, o.j, and o.k should be filled out already
	
	if (o.k == 0)
	{
		o.cReq = 1;
	}
	else
	{
		o.cReq = 2;
	}
	
	for (int i = o.k + 1; i < p; i++)
	{
		tuple<int, int, int, int> t = make_tuple(GEMM, i, o.i, o.k);
		o.connect.push_back(t);
	}
	o.finConnect = false;
}

void getGEMM(Op& o, int p)
{
	// o.id, o.i, o.j, and o.k should be filled out already
	
	o.cReq = 1;
	if (o.k > 0)
	{
		o.cReq++;
	}
	if (o.i != o.j)
	{
		o.cReq++;
	}
	
	if (o.i == o.j && o.j == o.k + 1)
	{
		tuple<int, int, int, int> t = make_tuple(POTF, 0, 0, o.k + 1);
		o.connect.push_back(t);
	}
	else if (o.j == o.k + 1)
	{
		tuple<int, int, int, int> t = make_tuple(TRSM, o.i, 0, o.k + 1);
		o.connect.push_back(t);
	}
	else
	{
		tuple<int, int, int, int> t = make_tuple(GEMM, o.i, o.j, o.k + 1);
		o.connect.push_back(t);
	}
}

void getFINISH(Op& o, int p)
{
	o.cReq = 1;
	o.finConnect = false;
}

Op getOp(tuple<int, int, int, int> t, int p)
{
	Op result;
	result.id = get<0>(t);
	result.i = get<1>(t);
	result.j = get<2>(t);
	result.k = get<3>(t);
	if (result.id == POTF)
	{
		getPOTF(result, p);
	}
	else if (result.id == TRSM)
	{
		getTRSM(result, p);
	}
	else if (result.id == GEMM)
	{
		getGEMM(result, p);
	}
	else if (result.id == FINISH)
	{
		getFINISH(result, p);
	}
	return result;
}

void operationDoer(OpArgs args)
{
	vector<vector<MatrixXd>>* A = args.A;
	queue<Op>* waitingOps = args.waitingOps;
	mutex* waitingMutex = args.waitingMutex;
	mutex* myMutex = args.myMutex;
	queue<Op>* myOps = args.myOps;
	int p = args.p;
	map<tuple<int, int, int, int>, int>* ops = args.ops;
	set<tuple<int, int, int, int>>* alreadyPut = args.alreadyPut;
	mutex* setMutex = args.setMutex;
	mutex* countMutex = args.countMutex;
	while (true)
	{
		lock_guard<mutex> lock(*myMutex);
		int size = myOps->size();
		delete &lock;
		if (size > 0)
		{
			lock_guard<mutex> lock2(*myMutex);
			Op curr = myOps->front();
			myOps->pop();
			delete &lock2;
			int i = curr.i;
			int j = curr.j;
			int k = curr.k;
			if (curr.id == POTF)
			{
				LLT<Ref<MatrixXd>> llt((*A)[k][k]);
			}
			else if (curr.id == TRSM)
			{
				(*A)[i][k] = (*A)[k][k].transpose().triangularView<Upper>().solve<OnTheRight>((*A)[i][k]);
			}
			else if (curr.id == GEMM)
			{
				(*A)[i][j] -= (*A)[i][k] * (*A)[j][k].transpose();
			}
			for (int i = 0; i < curr.connect.size(); i++)
			{
				lock_guard<mutex> lock3(*setMutex);
				bool putInAlready = alreadyPut->find(curr.connect[i]) != alreadyPut->end();
				delete &lock3;
				if (!putInAlready)
				{
					lock_guard<mutex> lock4(*waitingMutex);
					waitingOps->push(getOp(curr.connect[i], p));
					delete &lock4;
					lock_guard<mutex> lock5(*setMutex);
					alreadyPut->insert(curr.connect[i]);
					delete &lock5;
				}
				lock_guard<mutex> lock6(*countMutex);
				(*ops)[curr.connect[i]]++;
				delete &lock6;
			}
		}
	}
}

void cholesky(vector<vector<MatrixXd>>& A, int p, int b)
{
	map<tuple<int, int, int, int>, int> ops;
	queue<Op> waitingOps;
	queue<Op> readyOps[NUM_THREADS];
	mutex readyMutexes[NUM_THREADS];
	mutex waitingMutex;
	thread threads[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; i++)
	{
		OpArgs args;
		queue<Op> readyOp;
		readyOps[i] = readyOp;
		thread t(operationDoer, args);
		threads[i] = t;
		args.waitingOps = &waitingOps;
		args.myOps = &readyOps[i];
		args.waitingMutex = &waitingMutex;
		args.myMutex = &readyMutexes[i];
		args.A = &A;
		args.p = p;
	}
	int ind = 0;
	tuple<int, int, int, int> firstT(POTF, 0, 0, 0);
	Op firstOp = getOp(firstT, p);
	waitingOps.push(firstOp);
	while (true)
	{
		lock_guard<mutex> lock(waitingMutex);
		int size = waitingOps.size();
		delete &lock;
		if (size > 0)
		{
			lock_guard<mutex> lock2(waitingMutex);
			Op curr = waitingOps.front();
			waitingOps.pop();
			delete &lock2;
			tuple<int, int, int, int> tu = make_tuple(curr.id, curr.i, curr.j, curr.k);
			if (curr.cReq == ops[tu])
			{
				if (curr.id == FINISH)
				{
					break;
				}
				lock_guard<mutex> lock3(readyOps[ind % NUM_THREADS]);
				readyOps[ind % NUM_THREADS]->push(curr);
				delete &lock3;
				ind++;
			}
			else
			{
				lock_guard<mutex> lock4(waitingMutex);
				waitingOps.push(curr);
				delete &lock4;
			}
		}
	}
	for (int i = 0; i < NUM_THREADS; i++)
	{
		~threads[i];
	}
}

int main()
{
	// cout << "PROGRAM: START" << endl;
	
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
	
	// test();
	
	return 0;
}



































