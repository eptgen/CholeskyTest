#include "darvecholesky.h"
#include "unittests.h"
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <sys/time.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

double timeSubtract(timeval t1, timeval t2)
{
	long m1 = t1.tv_sec * 1000000 + t1.tv_usec;
	long m2 = t2.tv_sec * 1000000 + t2.tv_usec;
	return double(m1 - m2) / 1000000;
}

bool test(int n_thread)
{
	ifstream fin("tests/info.txt");
	int NUM_TESTS;
	fin >> NUM_TESTS;
	cout << NUM_TESTS << " tests" << endl;
	
	bool result = true;
	for (int i = 1; i <= NUM_TESTS; i++)
	{
		
		string fileName = "tests/";
		fileName.append(to_string(i));
		fileName.append(".txt");
		ifstream current(fileName);
		int size;
		int blockSize;
		int numBlocks;
		current >> size >> blockSize >> numBlocks;
		cout << "Test " << i << " (" << size << "x" << size << " matrix, " << blockSize << "x" << blockSize << " blocks)" << ": ";
		
		MatrixXd test(size, size);
		MatrixXd testCopy(size, size);
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				current >> test(j, k);
				testCopy(j, k) = test(j, k);
			}
		}
		
		timeval start;
		gettimeofday(&start, 0);
		cholesky(test, numBlocks, blockSize, n_thread);
		timeval end;
		gettimeofday(&end, 0);
		double secElapsed = timeSubtract(end, start);
		
		MatrixXd comp = test * test.transpose();
		/*
		cout << endl;
		cout << comp << endl;
		cout << testCopy << endl;
		*/
		
		// check for error < 1e-10
		bool shouldBreak = false;
		bool resultForCase = true;
		double error = 0;
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				double currentError = comp(j, k) - testCopy(j, k);
				if (currentError > 1e-10 || currentError < -1e-10)
				{
					shouldBreak = true;
					result = false;
					resultForCase = false;
					error = currentError;
					break;
				}
			}
			if (shouldBreak)
			{
				break;
			}
		}
		if (resultForCase)
		{
			cout << "SUCCESS (" << secElapsed << " s)" << endl;
		}
		else
		{
			cout << "FAIL (" << "error: " << error << ")" << endl;
		}
	}
	return result;
}





























