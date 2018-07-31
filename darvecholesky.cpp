#include "darvecholesky.h"
#include "dthread.cpp"
#include "unittests.h"
#include <string>
#include <functional>

using std::cout;
using std::endl;
using std::stoi;
using std::string;
using namespace std::placeholders;

void cholesky(MatrixXd& A, int p, int b, int n_thread)
{
	
	int nb = p;
	int n = p * b;
	// std::cout << "llt elapsed: " << duration_.count() << "\n";

	Block_matrix Ab(nb,nb);

	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			Ab(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
		}
	}

	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			assert(*(Ab(i,j)) == A.block(i*b,j*b,b,b));
		}
	}
	
	/*

	start = high_resolution_clock::now();
	// Calculate matrix product using blocks
	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			for (int k=0; k <nb; ++k) {
				*(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j));
			}
		}
	}
	end = high_resolution_clock::now();
	duration_ = duration_cast<milliseconds>(end - start);
	std::cout << "Block A*B elapsed: " << duration_.count() << "\n";
	*/

	/*
	// First test
	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			assert(*(Cb(i,j)) == C.block(i*b,j*b,b,b));
		}
	}
	*/

	/*
	// Copy back for testing purposes
	MatrixXd C0(n,n);
	for (int i=0; i<nb; ++i) {
		for (int j=0; j<nb; ++j) {
			MatrixXd & M = *(Cb(i,j));
			for (int i0=0; i0<b; ++i0) {
				for (int j0=0; j0<b; ++j0) {
					C0(i0+b*i,j0+b*j) = M(i0,j0);
				}
			}
		}
	}
	*/

	// Second test
	// assert(C == C0);

	// Re-doing calculation using task task_map

	// Create thread team
	Thread_team team(n_thread);

	// Task flow context
	struct Context {
		Task_flow potf;
		Task_flow trsm;
		Task_flow gemm;
		Context(Thread_team* a_tt, int nb) : potf(a_tt, nb, nb, nb), trsm(a_tt, nb, nb, nb), gemm(a_tt, nb, nb, nb) {}
	} ctx(&team, nb);
	
	// POTF
	
	auto l_potf = [&] (int m_id, int i, int k) {
		if (i == 1) {
			return true;
		}
		Eigen::LLT<Eigen::Ref<MatrixXd>> llt(*(Ab(k, k)));
		
		// printf("potf %d on thread %d\n", k, m_id);
		
		for (int l = k + 1; l < nb; l++)
		{
			ctx.trsm.decrement_wait_count({l, 0, k}, m_id);
		}
		if (k == nb - 1)
		{
			ctx.potf.decrement_wait_count({1, 0, nb - 1}, m_id);
		}
		return false;
	};
	
	auto l_potf_wait_count = [&] (int k) 
	{
		return (k == 0) ? 0 : 1;
	};
	
	ctx.potf = task_flow_init()
	.task_init( [=] (int3& idx, Task* a_tsk) {
		int wc = l_potf_wait_count(idx[2]);
		a_tsk->set_function(bind(l_potf, _1, idx[0], idx[2]));
		return wc;
	})
	.compute_on( [=] (int3& idx) {
		return idx[2] % n_thread;
	});
	ctx.potf.finish = {1, 0, nb - 1};
	
	// TRSM
	
	auto l_trsm = [&] (int m_id, int i, int k) {
		
		*(Ab(i, k)) = Ab(k, k)->transpose().triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(*(Ab(i, k)));
		
		// printf("trsm %d %d on thread %d\n", i, k, m_id);
		
		for (int l = k + 1; l < nb; l++)
		{
			if (l >= i)
			{
				// printf("trsm %d %d is decrementing %d %d %d\n", i, k, l, i, k);
				ctx.gemm.decrement_wait_count({l, i, k}, m_id);
			}
			else
			{
				// printf("trsm %d %d is decrementing %d %d %d\n", i, k, i, l, k);
				ctx.gemm.decrement_wait_count({i, l, k}, m_id);
			}
		}
		return false;
	};
	
	auto l_trsm_wait_count = [&] (int i, int k)
	{
		return (k == 0) ? 1 : 2;
	};
	
	ctx.trsm = task_flow_init()
	.task_init( [=] (int3& idx, Task* a_tsk) {
		int wc = l_trsm_wait_count(idx[0], idx[2]);
		a_tsk->set_function(bind(l_trsm, _1, idx[0], idx[2]));
		return wc;
	})
	.compute_on( [=] (int3& idx) {
		return (idx[0] + nb * idx[1]) % n_thread;
	});
	
	// GEMM
	
	auto l_gemm = [&] (int m_id, int i, int j, int k) {
		
		*(Ab(i, j)) -= *(Ab(i, k)) * Ab(j, k)->transpose();
		
		// printf("gemm %d %d %d on thread %d\n", i, j, k, m_id);
		
		if (i == j && j == k + 1)
		{
			ctx.potf.decrement_wait_count({0, 0, k + 1}, m_id);
		}
		else if (j == k + 1)
		{
			ctx.trsm.decrement_wait_count({i, 0, k + 1}, m_id);
		}
		else
		{
			ctx.gemm.decrement_wait_count({i, j, k + 1}, m_id);
		}
		return false;
	};
	
	auto l_gemm_wait_count = [&] (int i, int j, int k)
	{
		int nReq = 1;
		if (k > 0)
		{
			nReq++;
		}
		if (i != j)
		{
			nReq++;
		}
		return nReq;
	};
	
	ctx.gemm = task_flow_init()
	.task_init( [=] (int3& idx, Task* a_tsk) {
		int wc = l_gemm_wait_count(idx[0], idx[1], idx[2]);
		a_tsk->set_function(bind(l_gemm, _1, idx[0], idx[1], idx[2]));
		return wc;
	})
	.compute_on( [=] (int3& idx) {
		return (idx[0] + nb * idx[1]) % n_thread;
	});
	
	// Start team of threads
	team.start();

	// start = high_resolution_clock::now();

	// Create seed tasks and start
	/*
	for (int i=0; i<nb; ++i) {
		for (int j=0; j<nb; ++j) {
			ctx.init_mat.async({i,j,0});
		}
	}
	*/
	ctx.potf.async({0, 0, 0});

	// Wait for end of task queue execution
	team.join();

	// end = high_resolution_clock::now();
	// duration_ = duration_cast<milliseconds>(end - start);
	// std::cout << "CTXX GEMM elapsed: " << duration_.count() << "\n";

	for (int i = 0; i < nb; i++)
	{
		*(Ab(i, i)) = Ab(i, i)->triangularView<Eigen::Lower>();
	}
	for (int i = 0; i < nb; i++)
	{
		for (int j = i + 1; j < nb; j++)
		{
			*(Ab(i, j)) = MatrixXd::Zero(b, b);
		}
	}
	
	for (int i = 0; i < nb; i++)
	{
		for (int j = 0; j < nb; j++)
		{
			for (int k = 0; k < b; k++)
			{
				for (int l = 0; l < b; l++)
				{
					A(i * b + k, j * b + l) = (*(Ab(i, j)))(k, l);
				}
			}
		}
	}
	
	/*
	std::cout << "L:" << std::endl;
	std::cout << L << std::endl;
	
	std::cout << "The L That My Program Made:" << std::endl;
	std::cout << otherL << std::endl;
	*/
	
	/*
	// Test output
	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			assert(*(Cb(i,j)) == C.block(i*b,j*b,b,b));
		}
	}
	*/

	for (int i=0; i<nb; ++i) {
		for (int j=0; j <nb; ++j) {
			delete Ab(i,j);
		}
	}
}

int main(int argc, char* argv[])
{
	if (argc == 1)
	{
		test(4);
	}
	else
	{
		test(stoi(string(argv[1])));
	}

    return 0;
}
