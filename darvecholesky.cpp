#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <cassert>
#include <exception>
#include <stdexcept>

#include <random>

#include <list>
#include <queue>
#include <array>
#include <unordered_map>
#include <vector>

#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>
#include <sys/time.h>

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

using namespace std::chrono;

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::LLT;
using Eigen::Ref;
using Eigen::Upper;
using Eigen::OnTheRight;
using Eigen::Lower;

using std::min;
using std::max;

using std::list;
using std::vector;
using std::unordered_map;
using std::array;
using std::to_string;

using std::thread;
using std::mutex;
using std::bind;
using std::atomic_int;

typedef std::function<void()> Base_task;

const int POTF = 0;
const int GEMM = 1;
const int TRSM = 2;
const int FINISH = 3;

// Builds a logging message and outputs it in the destructor
class LogMessage {
public:
    LogMessage(const char * file, const char * function, int line)
    {
        os << file << ":" << line << " (" << function << ") ";
    }

    // output operator
    template<typename T>
    LogMessage & operator<<(const T & t)
    {
        os << t;
        return *this;
    }

    ~LogMessage()
    {
        os << "\n";
        std::cout << os.str();
        std::cout.flush();
    }
private:
    std::ostringstream os;
};

#if 0
#define LOG(out) do { \
  LogMessage(__FILE__,__func__,__LINE__) << out; \
} while (0)
#else
#define LOG(out)
#endif

void f_1()
{
}

void f_2()
{
}

void f_x(int * x)
{
    --(*x);
};

void f_count(std::atomic_int * c)
{
    ++(*c);
    std::this_thread::sleep_for(microseconds(20));
};

struct Fun_mv {
    MatrixXd *A;
    VectorXd *x, *y;
    void operator()()
    {
        (*y) = (*A) * (*x);
    }
};

struct Task : public Base_task {
    atomic_int wait_count; // Number of incoming edges/tasks

    bool m_delete = false;
    // whether the object should be deleted after running the function
    // to completion.

    float priority = 0.0; // task priority

    Task() {}

    Task(Base_task a_f, int a_wcount = 0, bool a_del = false, float a_priority = 0.0) :
        Base_task(a_f), m_delete(a_del), priority(a_priority)
    {
        wait_count.store(a_wcount);
    }

    ~Task() {};

    void operator=(Base_task a_tsk)
    {
        Base_task::operator=(a_tsk);
    }
};

// Task comparison based on their priorities
struct Task_comparison {
public:
    bool operator() (const Task* a_lhs, const Task* a_rhs) const
    {
        return (a_lhs->priority < a_rhs->priority);
    };
};

struct Thread_prio;
void spin(Thread_prio * a_thread);
struct Thread_team;

// Thread with priority queue management
struct Thread_prio {
    Thread_team * team;
    unsigned short m_id;

    std::priority_queue<Task*, vector<Task*>, Task_comparison> ready_queue;
    thread th;
    mutex mtx;
    std::condition_variable cv;
    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases
    std::atomic_bool m_stop;

    // Thread starts executing the function spin()
    void start()
    {
        m_empty.store(true);
        m_stop.store(false); // Used to return from spin()
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void spawn(Task * a_t)
    {
        LOG(m_id);
        std::lock_guard<std::mutex> lck(mtx);
        ready_queue.push(a_t); // Add task to queue
        m_empty.store(false);
        cv.notify_one(); // Wake up thread
    };

    Task* pop()
    {
        Task* tsk = ready_queue.top();
        ready_queue.pop();
        if (ready_queue.empty()) m_empty.store(true);
        return tsk;
    };

    // Set stop boolean to true so spin() can return
    void stop()
    {
        LOG(m_id);
        std::lock_guard<std::mutex> lck(mtx);
        m_stop.store(true);
        cv.notify_one(); // Wake up thread
    };

    // join() the thread
    void join()
    {
        stop();
        if (th.joinable()) {
            th.join();
        }
    }

    ~Thread_prio()
    {
        join();
    }
};

struct Thread_team : public vector<Thread_prio*> {
    vector<Thread_prio> v_thread;
    unsigned long n_thread_query = 16; // Optimization parameter

    Thread_team(const int n_thread) : v_thread(n_thread)
    {
        for (int i=0; i<n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
    }

    void start()
    {
        for (auto& th : v_thread) th.start();
    }

    void stop()
    {
        for (auto& th : v_thread) th.stop();
    }

    void join()
    {
        for (auto& th : v_thread) th.join();
    }

    void spawn(const int a_id, Task * a_task)
    {
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());
        int id_ = a_id;
        // Check if queue is empty
        if (!v_thread[a_id].m_empty.load()) {
            // Check whether other threads have empty queues
            const unsigned n_query = min(n_thread_query, v_thread.size());
            for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
                auto j = i%v_thread.size();
                if (v_thread[j].m_empty.load()) {
                    id_ = j;
                    break;
                }
            }
        }
        LOG("requested thread: " << a_id << " got " << id_);
        // Note that because we are not using locks, the queue may no longer be empty.
        // But this implementation is more efficient than using locks.
        v_thread[id_].spawn(a_task);
    }

    void steal(unsigned short a_id)
    {
        const unsigned n_query = min(n_thread_query, v_thread.size());
        for (unsigned long i=a_id+1; i<a_id+n_query; ++i) {
            auto j = i%v_thread.size();
            Thread_prio & thread_ = v_thread[j];
            if (!thread_.m_empty.load()) {
                std::unique_lock<std::mutex> lck(thread_.mtx);
                if (!thread_.ready_queue.empty()) {
                    Task * tsk = thread_.pop();
                    lck.unlock();
                    LOG(a_id << " from " << j);
                    v_thread[a_id].spawn(tsk);
                    break;
                }
            }
        }
    }
};

// Keep executing tasks until m_stop = true && queue is empty
void spin(Thread_prio * a_thread)
{
    LOG(a_thread->m_id);
    std::unique_lock<std::mutex> lck(a_thread->mtx);
    while (true) {
        while (!a_thread->ready_queue.empty()) {
            Task * tsk = a_thread->pop();
            lck.unlock();
            LOG(a_thread->m_id << " task()");
            (*tsk)();
            if (tsk->m_delete) delete tsk;
            lck.lock();
        }
        // Try to steal a task
        lck.unlock();
        a_thread->team->steal(a_thread->m_id);
        lck.lock();
        // Wait if queue is empty
        while (a_thread->ready_queue.empty()) {
            // Return if stop=true
            if (a_thread->m_stop.load()) {
                LOG(a_thread->m_id << " stop");
                return;
            }
            LOG(a_thread->m_id << " wait");
            a_thread->cv.wait(lck);
        }
    }
};

namespace hash_array {

inline void hash_combine(std::size_t& seed, int const& v)
{
    seed ^= std::hash<int>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct hash {
    size_t
    operator()(array<int,4> const& key) const
    {
        size_t seed = 0;
        hash_combine(seed, key[0]);
        hash_combine(seed, key[1]);
        hash_combine(seed, key[2]);
		hash_combine(seed, key[3]);
        return seed;
    }
};

}

typedef array<int,4> int4;

struct Task_graph {
    Thread_team * team = nullptr;

    unordered_map< int4, Task*, hash_array::hash > graph;
    mutex mtx_graph, mtx_quiescence;
    atomic_int n_active_task;
    std::condition_variable cond_quiescence;

    bool quiescence = false;
    // boolean: all tasks have been posted and quiescence has been reached

    virtual ~Task_graph() {};

    // How to initialize a task
    virtual void initialize_task(int4&,Task*) = 0;

    // Mapping from int4 task index to thread id
    virtual int task_map(int4&) = 0;

    // Find a task in the graph and return pointer
    Task * find_task(int4&);

    // spawn task with index
    void async(int4);
    // spawn with index and task
    void async(int4, Task*);

    // Decrement the dependency counter and spawn task if ready
    void decrement_wait_count(int4, int4);

    // Returns when all tasks have been posted
    void wait()
    {
        std::unique_lock<std::mutex> lck(mtx_quiescence);
        while (!quiescence) cond_quiescence.wait(lck);
    }
};

Task * Task_graph::find_task(int4 & idx)
{
    Task * t_ = nullptr;

    std::lock_guard<std::mutex> lck(mtx_graph);
    auto tsk = graph.find(idx);

    // Task exists
    if (tsk != graph.end()) {
        t_ = tsk->second;
    }
    else {
        // Task does not exist; create it
        t_ = new Task;
        graph[idx] = t_; // Insert in graph
        initialize_task(idx,t_);
        ++ n_active_task; // Increment counter
    }

    assert(t_ != nullptr);

    return t_;
}

void Task_graph::async(int4 idx, Task * a_tsk)
{
    LOG("spawn task " << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3]);
	std::string text = "async ";
	text += to_string(idx[0]);
	text += " ";
	text += to_string(idx[1]);
	text += " ";
	text += to_string(idx[2]);
	text += " ";
	text += to_string(idx[3]);
	// std::cout << text << std::endl;

    // Delete entry in graph
    {
        std::lock_guard<std::mutex> lck(mtx_graph);
        if (graph.find(idx) == graph.end())
		{
			return;
		}
        graph.erase(idx);
    }

    team->spawn(task_map(idx), a_tsk);

    -- n_active_task; // Decrement counter

    assert(n_active_task.load() >= 0);

    // Signal if quiescence has been reached
    if (n_active_task.load() == 0) {
        std::unique_lock<std::mutex> lck(mtx_quiescence);
        quiescence = true;
        cond_quiescence.notify_one(); // Notify waiting thread
    }
}

void Task_graph::async(int4 idx)
{
    Task * t_ = find_task(idx);
    async(idx, t_);
}

void Task_graph::decrement_wait_count(int4 idx, int4 src)
{
    Task * t_ = find_task(idx);
	
	int before = t_->wait_count;
	
    // Decrement counter
    --( t_->wait_count );
	
	int after = t_->wait_count;
	
	
	std::string str = "decrementing ";
	str += to_string(idx[0]);
	str += " ";
	str += to_string(idx[1]);
	str += " ";
	str += to_string(idx[2]);
	str += " ";
	str += to_string(idx[3]);
	str += " from ";
	str += to_string(before);
	str += " to ";
	str += to_string(after);
	str += "\n";
	str += "Source: ";
	str += to_string(src[0]);
	str += " ";
	str += to_string(src[1]);
	str += " ";
	str += to_string(src[2]);
	str += " ";
	str += to_string(src[3]);
	str += "\n";
	// std::cout << str;

    assert(t_->wait_count.load() >= 0);

    if (t_->wait_count.load() == 0) { // task is ready to run
		std::string text = "async ";
		text += to_string(idx[0]);
		text += " ";
		text += to_string(idx[1]);
		text += " ";
		text += to_string(idx[2]);
		text += " ";
		text += to_string(idx[3]);
		text += "\n";
		text += "Source: ";
		text += to_string(src[0]);
		text += " ";
		text += to_string(src[1]);
		text += " ";
		text += to_string(src[2]);
		text += " ";
		text += to_string(src[3]);
		text += "\n";
		// std::cout << text;
        async(idx, t_);
    }
}

struct Block_matrix : vector<MatrixXd*> {
    int row, col;
    Block_matrix() {};
    Block_matrix(int a_row, int a_col) : vector<MatrixXd*>(a_row*a_col),
        row(a_row), col(a_col) {};

    void resize(int a_row, int a_col)
    {
        vector<MatrixXd*>::resize(a_row*a_col);
        row = a_row;
        col = a_col;
    }

    MatrixXd* & operator()(int i, int j)
    {
        assert(i>=0 && i<row);
        assert(j>=0 && j<col);
        return operator[](i + j*row);
    }
};

struct Chol_graph: public Task_graph {
	int size_id = -1;
    int size_i = -1;
    int size_j = -1;
    int size_k = -1;
    Block_matrix A;

    void initialize_task(int4&, Task*);

    int task_map(int4 & idx)
    {
        return ( ( idx[0] + size_i * idx[1] ) % ( team->v_thread.size() ) );
    }

    void start();
};

void fun_chol(Chol_graph * m_g, int id, int i, int j, int k)
{
	assert(id>=0 && id < m_g->size_id);
    assert(i>=0 && i<m_g->size_i);
    assert(j>=0 && j<m_g->size_j);
    assert(k>=0 && k<m_g->size_k);
    LOG(i << " " << j << " " << k);

	if (id == POTF)
	{
		LLT<Ref<MatrixXd>> llt(*m_g->A(k, k));
		for (int l = k + 1; l < m_g->size_i; l++)
		{
			m_g->decrement_wait_count({TRSM, l, 0, k}, {id, i, j, k});
		}
	}
	else if (id == TRSM)
	{
		*m_g->A(i, k) = m_g->A(k, k)->transpose().triangularView<Upper>().solve<OnTheRight>(*m_g->A(i, k));
		for (int l = k + 1; l < m_g->size_i; l++)
		{
			if (l >= i)
			{
				m_g->decrement_wait_count({GEMM, l, i, k}, {id, i, j, k});
			}
			else
			{
				m_g->decrement_wait_count({GEMM, i, l, k}, {id, i, j, k});
			}
		}
	}
	else if (id == GEMM)
	{
		*m_g->A(i, j) -= *m_g->A(i, k) * m_g->A(j, k)->transpose();
		if (i == j && j == k + 1)
		{
			m_g->decrement_wait_count({POTF, 0, 0, k + 1}, {id, i, j, k});
		}
		else if (j == k + 1)
		{
			m_g->decrement_wait_count({TRSM, i, 0, k + 1}, {id, i, j, k});
		}
		else
		{
			m_g->decrement_wait_count({GEMM, i, j, k + 1}, {id, i, j, k});
		}
	}
	// std::cout << id << " " << i << " " << j << " " << k << std::endl;
}

void Chol_graph::initialize_task(int4 & idx, Task* a_tsk)
{
    assert(a_tsk != nullptr);
	int id = idx[0];
	int i = idx[1];
	int j = idx[2];
	int k = idx[3];
	
	if (id == POTF)
	{
		if (k == 0)
		{
			a_tsk->wait_count.store(0);
		}
		else
		{
			a_tsk->wait_count.store(1);
		}
	}
	else if (id == TRSM)
	{
		if (k == 0)
		{
			a_tsk->wait_count.store(1);
		}
		else
		{
			a_tsk->wait_count.store(2);
		}
	}
	else if (id == GEMM)
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
		/*
		if (i == 2 && j == 1 && k == 0)
		{
			std::cout << "nReq: " << nReq << std::endl;
		}
		*/
		a_tsk->wait_count.store(nReq);
	}
	
    a_tsk->m_delete = true; // free memory after execution
    (*a_tsk) = bind(fun_chol,this,idx[0],idx[1],idx[2], idx[3]);
}

void Chol_graph::start()
{
    assert(size_i>0);
    assert(size_j>0);
    assert(size_k>0);
	assert(size_id>0);
    assert(A.size() == static_cast<unsigned long>(size_i*size_k));
    assert(team != nullptr);

    n_active_task.store(0); // Active task counter

    async({POTF, 0, 0, 0});
}

void test();

int main(void)
{

    try {
        test();
    }
    catch (std::exception& a_e) {
        std::cout << a_e.what() << '\n';
    }

    return 0;
}

double timeSubtract(timeval t1, timeval t2)
{
	long m1 = t1.tv_sec * 1000000 + t1.tv_usec;
	long m2 = t2.tv_sec * 1000000 + t2.tv_usec;
	return double(m1 - m2) / 1000000;
}

void test()
{

    {
        Task t1( f_1 );
        assert(t1.wait_count.load() == 0);

        Task t2( f_2, 4 );
        assert(t2.wait_count.load() == 4);

        t1();
        t2();
    }

    {
        int x = 2;
        assert(x == 2);

        Task t(bind(f_x, &x));
        t();

        assert(x == 1);
    }

    {
        MatrixXd A(2,2);
        A << 1,1,2,-2;
        assert(A(0,0) == 1);
        assert(A(0,1) == 1);
        assert(A(1,0) == 2);
        assert(A(1,1) == -2);
        VectorXd x(2);
        x << 3,2;
        assert(x(0) == 3);
        assert(x(1) == 2);
        VectorXd y(2);

        // Demonstrating a function with arguments without bind()
        Fun_mv f_mv;
        f_mv.A = &A;
        f_mv.x = &x;
        f_mv.y = &y;

        Task t(f_mv);
        t();

        assert(y(0) == 5);
        assert(y(1) == 2);
    }

    {
        const int n_thread = 4;
        const int max_count = 10;

        // Create thread team
        Thread_team team(n_thread);
        vector<std::atomic_int> counter(n_thread);

        for(auto & c : counter) {
            c.store(0);
        }

        vector<Task> tsk(n_thread);
        for(int nt=0; nt<n_thread; ++nt) {
            tsk[nt] = bind(f_count, &counter[nt]);
        }

#ifdef PROFILER
        ProfilerStart("ctxx.pprof");
#endif
        {
            team.start();
#ifdef PROFILER
            auto start = high_resolution_clock::now();
#endif
            for(int it=0; it < max_count; ++it) {
                for(int nt=0; nt<n_thread; ++nt) {
                    team.spawn(0, &tsk[nt]);
                }
            }
            team.join();
#ifdef PROFILER
            auto end = high_resolution_clock::now();
            auto duration_ = duration_cast<milliseconds>(end - start);
            std::cout << "Elapsed: " << duration_.count() << "\n";
#endif
        }

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt].load() == max_count);
        }

        for(auto & c : counter) {
            c.store(0);
        }

        {
            team.start();
#ifdef PROFILER
            auto start = high_resolution_clock::now();
#endif
            for(int it=0; it < max_count; ++it) {
                for(int nt=0; nt<n_thread; ++nt) {
                    team.spawn(nt, &tsk[nt]);
                }
            }
            team.join();
#ifdef PROFILER
            auto end = high_resolution_clock::now();
            auto duration_ = duration_cast<milliseconds>(end - start);
            std::cout << "Elapsed: " << duration_.count() << "\n";
#endif
        }

#ifdef PROFILER
        ProfilerStop();
#endif

        for(int nt=0; nt<n_thread; ++nt) {
            assert(counter[nt].load() == max_count);
        }
    }

	// Initialize GEMM matrices
	std::mt19937 mers_rand;
	// Seed the engine
	mers_rand.seed(2018);
	
	timeval start;
	gettimeofday(&start, NULL);
    for (int i=0; i<100; ++i) {

        if (i%10 == 0) printf("i %d\n",i);

        const int nb = 8; // number of blocks
        const int b = 4; // size of blocks

        const int n_thread = 1; // Number of threads to use

        const int n = b*nb; // matrix size

        MatrixXd A(n,n);

        LOG("matrix");

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                A(i,j) = int(mers_rand()) % 20 - 10;
            }
        }
		// std::cout << "A: " << std::endl << A << std::endl;
		A = A * A.transpose() + MatrixXd::Identity(n, n) * 0.0005;
		// std::cout << "A: " << std::endl << A << std::endl;

        auto start = high_resolution_clock::now();
        MatrixXd L = A.llt().matrixL();
        //printf("First entry in C: %g\n",C(0,0));
        auto end = high_resolution_clock::now();
        auto duration_ = duration_cast<milliseconds>(end - start);

        LOG("init");

        Chol_graph chol_g;
		chol_g.size_id = 4;
        chol_g.size_i = nb;
        chol_g.size_j = nb;
        chol_g.size_k = nb;
        chol_g.A.resize(nb,nb);

        LOG("graph");

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                chol_g.A(i,j) = new MatrixXd(A.block(i*b,j*b,b,b));
            }
        }

        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
                assert(*chol_g.A(i,j) == A.block(i*b,j*b,b,b));
            }
        }

        // Create thread team
        Thread_team team(n_thread);

        chol_g.team = &team;

        // Start team of threads
        team.start();

        start = high_resolution_clock::now();

        LOG("start");
        // Create seed tasks in graph and start
        chol_g.start();

        LOG("wait");
        // Wait for all tasks to be posted
        chol_g.wait();

        LOG("join");
        // Wait for end of task queue execution
        team.join();

        end = high_resolution_clock::now();
        duration_ = duration_cast<milliseconds>(end - start);
        //std::cout << "CTXX GEMM elapsed: " << duration_.count() << "\n";

        LOG("test");
		
		
		for (int i = 0; i < nb; i++)
		{
			*chol_g.A(i, i) = chol_g.A(i, i)->triangularView<Lower>();
		}
		for (int i = 0; i < nb; i++)
		{
			for (int j = i + 1; j < nb; j++)
			{
				*chol_g.A(i, j) = MatrixXd::Zero(b, b);
			}
		}
		
		MatrixXd B(n, n);
		for (int i = 0; i < nb; i++)
		{
			for (int j = 0; j < nb; j++)
			{
				for (int k = 0; k < b; k++)
				{
					for (int l = 0; l < b; l++)
					{
						B(i * b + k, j * b + l) = (*chol_g.A(i, j))(k, l);
					}
				}
			}
		}
		
		// std::cout << "Real L: " << std::endl << L << std::endl;
		// std::cout << "Test L: " << std::endl << B << std::endl;

        // Test output
        for (int i=0; i<nb; ++i) {
            for (int j=0; j <nb; ++j) {
				MatrixXd error = *chol_g.A(i,j) - L.block(i*b,j*b,b,b);
                for (int k = 0; k < b; k++)
				{
					for (int l = 0; l < b; l++)
					{
						if (error(k, l) > 1e-8)
						{
							std::cout << error(k, l) << std::endl;
						}
					}
				}
            }
        }
    }
	timeval end;
	gettimeofday(&end, NULL);
	double secondsElapsed = timeSubtract(end, start);
	std::cout << "total time: " << secondsElapsed << " s" << std::endl;
}
