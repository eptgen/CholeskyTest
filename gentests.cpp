#include <iostream>
#include <fstream>
#include <random>
#include <time.h>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
	srand(time(NULL));
	
	ifstream fin("testsizes.txt");
	
	int NUM_TESTS;
	fin >> NUM_TESTS;
	
	cout << "GENERATING " << NUM_TESTS << " TESTS" << endl;
	
	ofstream fout("tests/info.txt");
	fout << NUM_TESTS << endl;
	
	for (int i = 0; i < NUM_TESTS; i++)
	{
		cout << "GENERATING #" << (i + 1) << "... ";
		int testSize;
		int blockSize;
		int numBlocks;
		float smallNumber;
		fin >> testSize >> blockSize >> numBlocks >> smallNumber;
		MatrixXd test(testSize, testSize);
		for (int i = 0; i < testSize; i++)
		{
			for (int j = 0; j < testSize; j++)
			{
				test(i, j) = rand() % 20 - 10;
			}
		}
		test = test * test.transpose() + smallNumber * MatrixXd::Identity(testSize, testSize);
		
		string fileName = "tests/";
		fileName.append(to_string(i + 1));
		fileName.append(".txt");
		ofstream fout(fileName);
		fout << testSize << " " << blockSize << " " << numBlocks << " " << endl;
		for (int i = 0; i < testSize; i++)
		{
			for (int j = 0; j < testSize; j++)
			{
				fout << test(i, j) << " ";
			}
			fout << endl;
		}
		cout << "done" << endl;
	}
}

































