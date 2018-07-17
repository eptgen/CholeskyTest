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

#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

using namespace std::chrono;

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::min;
using std::max;

using std::list;
using std::vector;
using std::unordered_map;
using std::array;

using std::thread;
using std::mutex;
using std::bind;
using std::atomic_int;

typedef std::function<void()> Base_task;

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
    int m_wait_count; // Number of incoming edges/tasks

    float priority = 0.0; // task priority

    bool m_delete = true;
    // whether the object should be deleted after running the function
    // to completion.

	mutex mtx;
	
    Task() {}

    Task(Base_task a_f, int a_wcount = 0, float a_priority = 0.0, bool a_del = true) :
        Base_task(a_f), m_wait_count(a_wcount), priority(a_priority), m_delete(a_del)
    {}

    ~Task() {};

    void init(Base_task a_f, int a_wcount = 0, float a_priority = 0.0, bool a_del = true)
    {
        Base_task::operator=(a_f);
        m_wait_count = a_wcount;
        priority = a_priority;
        m_delete = a_del;
    }

    void operator=(Base_task a_f)
    {
        Base_task::operator=(a_f);
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
    operator()(array<int,3> const& key) const
    {
        size_t seed = 0;
        hash_combine(seed, key[0]);
        hash_combine(seed, key[1]);
        hash_combine(seed, key[2]);
        return seed;
    }
};

}

typedef array<int,3> int3;

struct Task_flow {

    typedef std::function<void(int3&,Task*)> Init_task;
    typedef std::function<int(int3&)> Map_task;

    Thread_team * team = nullptr;

    struct Init {
        // How to initialize a task
        Init_task init;
        // Mapping from task index to thread id
        Map_task map;

        Init& task_init(Init_task a_init)
        {
            init = a_init;
            return *this;
        }

        Init& compute_on(Map_task a_map)
        {
            map = a_map;
            return *this;
        }
    } m_task;

    Task_flow& operator=(Init & a_setup)
    {
        m_task.init = a_setup.init;
        m_task.map = a_setup.map;
        return *this;
    }

    unordered_map< int3, Task*, hash_array::hash > task_map;

    mutex mtx_graph, mtx_quiescence;
    std::condition_variable cond_quiescence;

    bool quiescence = false;
    // true = all tasks have been posted and quiescence has been reached

    int n_task_in_graph = 0;

    Task_flow(Thread_team * a_team) : team(a_team) {}

    virtual ~Task_flow() {}

    // Find a task in the task_map and return pointer
    Task * find_task(int3&);

    // spawn task with index
    void async(int3);
    // spawn with index and task
    void async(int3, Task*);

    // Decrement the dependency counter and spawn task if ready
    void decrement_wait_count(int3);

    // Returns when all tasks have been posted
    void wait()
    {
        std::unique_lock<std::mutex> lck(mtx_quiescence);
        while (!quiescence) cond_quiescence.wait(lck);
    }
};

Task_flow::Init task_flow_init()
{
    return Task_flow::Init();
}

Task * Task_flow::find_task(int3 & idx)
{
    Task * t_ = nullptr;

    std::unique_lock<std::mutex> lck(mtx_graph);

    auto tsk = task_map.find(idx);

    // Task exists
    if (tsk != task_map.end()) {
        t_ = tsk->second;
    }
    else {
        // Task does not exist; create it
        t_ = new Task;
        task_map[idx] = t_; // Insert in task_map

        m_task.init(idx,t_); // Initialize

        ++n_task_in_graph; // Increment counter
    }

    lck.unlock();

    assert(t_ != nullptr);

    return t_;
}

void Task_flow::async(int3 idx, Task * a_tsk)
{
    team->spawn(/*task map*/ m_task.map(idx), a_tsk);

    // Delete entry in task_map
    std::unique_lock<std::mutex> lck(mtx_graph);

    assert(task_map.find(idx) != task_map.end());
    task_map.erase(idx);

    -- n_task_in_graph; // Decrement counter
    assert(n_task_in_graph >= 0);

    // Signal if quiescence has been reached
    if (n_task_in_graph == 0) {
        lck.unlock();
        std::unique_lock<std::mutex> lck(mtx_quiescence);
        quiescence = true;
        cond_quiescence.notify_one(); // Notify waiting thread
    }
}

void Task_flow::async(int3 idx)
{
    Task * t_ = find_task(idx);
    async(idx, t_);
}

void Task_flow::decrement_wait_count(int3 idx)
{
    Task * t_ = find_task(idx);

    // Decrement counter
    std::unique_lock<std::mutex> lck(t_->mtx);
	--(t_->m_wait_count);
	assert(t_->m_wait_count >= 0);

    if (t_->m_wait_count == 0) { // task is ready to run
		lck.unlock();
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

void test()
{

    {
        Task t1( f_1 );
        assert(t1.m_wait_count == 0);

        Task t2( f_2, 4 );
        assert(t2.m_wait_count == 4);

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
            tsk[nt].init(bind(f_count, &counter[nt]), 0, 0., false);
        }

        {
            team.start();
#ifdef PROFILER
            auto start = high_resolution_clock::now();
#endif
            for(int it=0; it < max_count; ++it) {
                for(int nt=0; nt<n_thread; ++nt) {
                    team.spawn(0, &tsk[nt]);
                    // spawn @ 0
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
                    // spawn @ nt
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
    }

    {
        const int nb = 32; // number of blocks
        const int b = 16;  // size of blocks

        const int n_thread = 128; // number of threads to use

        const int n = b*nb; // matrix size

        MatrixXd A(n,n);

        // Initialize GEMM matrices
        std::mt19937 mers_rand;
        // Seed the engine
        mers_rand.seed(2018);

        for (int i=0; i<n; ++i) {
            for (int j=0; j<n; ++j) {
                A(i,j) = int(mers_rand()) % 20 - 10;
            }
        }
		// std::cout << "A: " << std::endl << A << std::endl;
		A = A * A.transpose() + MatrixXd::Identity(n, n) * 0.0005;
		
        auto start = high_resolution_clock::now();
        MatrixXd L = A.llt().matrixL();
        auto end = high_resolution_clock::now();
        auto duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "llt elapsed: " << duration_.count() << "\n";

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
            Context(Thread_team* a_tt) : potf(a_tt), trsm(a_tt), gemm(a_tt) {}
        } ctx(&team);
		
		// POTF
		
		auto l_potf = [&] (int k) {
			Eigen::LLT<Eigen::Ref<MatrixXd>> llt(*(Ab(k, k)));
			for (int l = k + 1; l < nb; l++)
			{
				ctx.trsm.decrement_wait_count({l, 0, k});
			}
		};
		
		auto l_potf_wait_count = [&] (int k) 
		{
			return (k == 0) ? 0 : 1;
		};
		
		ctx.potf = task_flow_init()
		.task_init( [=] (int3& idx, Task* a_tsk) {
			int wc = l_potf_wait_count(idx[2]);
			a_tsk->init(bind(l_potf, idx[2]), wc);
		})
		.compute_on( [=] (int3& idx) {
			return idx[2] % n_thread;
		});
		
		// TRSM
		
		auto l_trsm = [&] (int i, int k) {
			
			*(Ab(i, k)) = Ab(k, k)->transpose().triangularView<Eigen::Upper>().solve<Eigen::OnTheRight>(*(Ab(i, k)));
			for (int l = k + 1; l < nb; l++)
			{
				if (l >= i)
				{
					ctx.gemm.decrement_wait_count({l, i, k});
				}
				else
				{
					ctx.gemm.decrement_wait_count({i, l, k});
				}
			}
		};
		
		auto l_trsm_wait_count = [&] (int i, int k)
		{
			return (k == 0) ? 1 : 2;
		};
		
		ctx.trsm = task_flow_init()
		.task_init( [=] (int3& idx, Task* a_tsk) {
			int wc = l_trsm_wait_count(idx[0], idx[2]);
			a_tsk->init(bind(l_trsm, idx[0], idx[2]), wc);
		})
		.compute_on( [=] (int3& idx) {
			return (idx[0] + nb * idx[1]) % n_thread;
		});
		
		// GEMM
		
		auto l_gemm = [&] (int i, int j, int k) {
			
			*(Ab(i, j)) -= *(Ab(i, k)) * Ab(j, k)->transpose();
			if (i == j && j == k + 1)
			{
				ctx.potf.decrement_wait_count({0, 0, k + 1});
			}
			else if (j == k + 1)
			{
				ctx.trsm.decrement_wait_count({i, 0, k + 1});
			}
			else
			{
				ctx.gemm.decrement_wait_count({i, j, k + 1});
			}
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
			a_tsk->init(bind(l_gemm, idx[0], idx[1], idx[2]), wc);
		})
		.compute_on( [=] (int3& idx) {
			return (idx[0] + nb * idx[1]) % n_thread;
		});
		
		/*
        // Init task
        auto l_init = [&] (int i, int j) {
            *(Ab(i,j)) = A.block(i*b,j*b,b,b);
            for (int k=0; k<nb; ++k) {
                ctx.gemm_g.decrement_wait_count({i,k,0});
                ctx.gemm_g.decrement_wait_count({k,j,0});
            }
        };

        ctx.init_mat = task_flow_init()
        .task_init( [=] (int3& idx, Task* a_tsk) {
            a_tsk->init( bind(l_init,idx[0],idx[1]),
                           0);
        })
        .compute_on([=] (int3& idx) {
            return ( ( idx[0] + nb * idx[1] ) % n_thread );
        });

        // GEMM task
        auto l_gemm = [&] (int i, int j, int k) {
            *(Cb(i,j)) += *(Ab(i,k)) * *(Bb(k,j)); // GEMM
            if (k < nb - 1) {
                ctx.gemm_g.decrement_wait_count({i,j,k+1});
            }
        };

        auto l_gemm_wait_count = [&] (int k) {
            return ( k == 0 ? 2*nb : 1);
        };

        ctx.gemm_g = task_flow_init()
        .task_init( [=] (int3& idx, Task* a_tsk) {
            int wait_count = l_gemm_wait_count( idx[2]);
            a_tsk->init( bind(l_gemm,idx[0],idx[1],idx[2]),
                         wait_count);
        })
        .compute_on([=] (int3& idx) {
            return ( ( idx[0] + nb * idx[1] ) % n_thread );
        });

		*/
		
        // Start team of threads
        team.start();

#ifdef PROFILER
        ProfilerStart("ctxx.pprof");
#endif
        start = high_resolution_clock::now();

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

#ifdef PROFILER
        ProfilerStop();
#endif

        end = high_resolution_clock::now();
        duration_ = duration_cast<milliseconds>(end - start);
        std::cout << "CTXX GEMM elapsed: " << duration_.count() << "\n";

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
		
		MatrixXd otherL(n, n);
		for (int i = 0; i < nb; i++)
		{
			for (int j = 0; j < nb; j++)
			{
				for (int k = 0; k < b; k++)
				{
					for (int l = 0; l < b; l++)
					{
						otherL(i * b + k, j * b + l) = (*(Ab(i, j)))(k, l);
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
}
