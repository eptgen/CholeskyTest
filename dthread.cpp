#ifndef MAIN_CPP
#define MAIN_CPP

#include <cstdio>
#include <string>
#include <iostream>
#include <sstream>
#include <cassert>
#include <exception>
#include <stdexcept>
#include <functional>

#include <random>
#include <time.h>
#include <sys/time.h>

#include <list>
#include <deque>
#include <array>
#include <unordered_map>
#include <vector>

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
using std::atomic_bool;
using std::atomic;

typedef std::function<bool(int)> Base_task;

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

double timeSub(timeval t1, timeval t2)
{
	long m1 = t1.tv_sec * 1000000 + t1.tv_usec;
	long m2 = t2.tv_sec * 1000000 + t2.tv_usec;
	return double(m1 - m2) / 1000000;
}

struct Task : public Base_task {
    int m_wait_count; // Number of incoming edges/tasks

    float priority = 0.0; // task priority

    bool m_delete = true;
    // whether the object should be deleted after running the function
    // to completion.

    mutex mtx; // Protects concurrent access to m_wait_count

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

int NO_REQU = -1;
Task taskk;
Task* NO_RESP = &taskk;

bool compare_and_swap(atomic_int* ptr, int oldval, int newval) {
	int old_reg_val = ptr->load();
	bool result = old_reg_val == oldval;
	if (result) {
		// std::cout << "haha now you freeze\n";
		ptr->store(newval);
	}
	return result;
}

// Thread with priority queue management
struct Thread_prio {
    Thread_team * team;
    unsigned short m_id;

    std::deque<Task*> ready_queue;
    thread th;
    std::condition_variable cv;
    std::atomic_bool m_empty;
    // For optimization to avoid testing ready_queue.empty() in some cases
    std::atomic_bool m_stop;

    // Thread starts executing the function spin()
    void start()
    {
		// std::cout << "starting a thread\n";
        m_empty.store(true);
		transfer.store(NO_RESP);
		request.store(NO_REQU);
        m_stop.store(false); // Used to return from spin()
		status.store(false);
        th = thread(spin, this); // Execute tasks in queue
    };

    // Add new task to queue
    void spawn(Task * a_t)
    {
        LOG(m_id);
		// std::cout << "spawning a task\n";
        // std::lock_guard<std::mutex> lck(mtx);
		/*
		timeval start;
		timeval end;
		gettimeofday(&start, 0);
		std::string str1 = "putting a task in ";
		str1 += std::to_string(m_id);
		str1 += "\n";
		std::cout << str1;
		*/
        ready_queue.push_back(a_t); // Add task to queue
		// printf("pushing. new size: %d\n", ready_queue.size());
		/*
		gettimeofday(&end, 0);
		double secElapsed = timeSub(end, start);
		std::string str = "Spawning task (";
		str += std::to_string(secElapsed);
		str += " s)\n";
		std::string str = "Number of tasks in ";
		str += std::to_string(m_id);
		str += ": ";
		str += std::to_string(ready_queue.size());
		str += "\n";
		std::cout << str;
		*/
        m_empty.store(false);
        cv.notify_one(); // Wake up thread
		update_status();
    };

    Task* pop_bottom()
    {
        Task* tsk = ready_queue.back();
        ready_queue.pop_back();
        if (ready_queue.empty()) m_empty.store(true);
        return tsk;
    };
	
	Task* pop_top()
	{
		Task* tsk = ready_queue.front();
		ready_queue.pop_front();
		if (ready_queue.empty()) m_empty.store(true);
		return tsk;
	}

    // Set stop boolean to true so spin() can return
    void stop()
    {
        LOG(m_id);
        // std::lock_guard<std::mutex> lck(mtx);
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
	
	atomic_bool status;
	atomic<Task*> transfer;
	atomic_int request;
	
	void acquire();
	
	void update_status() {
		status.store(!ready_queue.empty());
	}
	
	void communicate();
};

struct Thread_team : public vector<Thread_prio*> {
    vector<Thread_prio> v_thread;
    unsigned long n_thread_query = 16; // Optimization parameter
	atomic_bool endMe;
	
    Thread_team(const int n_thread) : v_thread(n_thread)
    {
		srand(time(NULL));
		endMe.store(false);
        for (int i=0; i<n_thread; ++i) {
            v_thread[i].team = this;
            v_thread[i].m_id = static_cast<unsigned short>(i);
        }
    }
	
	Thread_prio* getThread(int i) {
		return &(v_thread[i]);
	}
	
	Thread_prio* random_thread(unsigned short except)
	{
		// printf("size of v_thread:%d\n", v_thread.size());
		unsigned short id = rand() % (v_thread.size() - 1);
		if (id >= except) {
			id++;
		}
		// printf("spitting out id=%d\n", id);
		return &(v_thread[id]);
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
		// while (!(endMe.load())) {}
        for (auto& th : v_thread) th.join();
    }

    void spawn(const int a_id, Task * a_task, bool shouldStop = false)
    {
		/*
        assert(a_id >= 0 && static_cast<unsigned long>(a_id) < v_thread.size());
		if (a_task == nullptr) {
			printf("oops");
		}
		
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
		// if (shouldStop) {
		// 	std::cout << "attempting to stop\n";
		// 	endMe.store(true);
		// }
		*/
		
		v_thread[a_id].spawn(a_task);
    }
	
	/*
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
	*/
};

void Thread_prio::acquire() {
	while (!(team->endMe.load()) && ready_queue.empty()) {
		/*
		if (m_id != 0) {
			printf("hi %d\n", m_id);
		}
		*/
		// cout << sdjfsdfsf
		// cout << sldjfs << dslfksj << dlsfdls
		transfer.store(NO_RESP);
		Thread_prio* thh = team->random_thread(m_id);
		/*
		if (m_id != 0) {
			printf("requesting from %d and the current thread is %d\n", thh->m_id, m_id);
		}
		*/
		if (thh->status && compare_and_swap(&(thh->request), NO_REQU, m_id)) {
			while (transfer.load() == NO_RESP && !(team->endMe.load()) && ready_queue.empty()) {
				/*
				if (m_id != 0) {
					printf("hi %d\n", m_id);
				}
				*/
				communicate();
			}
			if (transfer.load() != nullptr) {
				// std::cout << "stealing is a crime\n";
				spawn(transfer.load());
				request.store(NO_REQU);
				return;
			}
		}
		/*
		std::string str = std::to_string(m_id);
		str += " ";
		str += std::to_string(ready_queue.size());
		str += "\n";
		std::cout << str;
		*/
		communicate();
	}
}

void Thread_prio::communicate() {
	int j = request.load();
	if (j == NO_REQU) return;
	if (ready_queue.empty()) {
		// std::printf("%d is empty\n", m_id);
		team->getThread(j)->transfer.store(nullptr);
	} else {
		// printf("%d is transferring to %d\n", m_id, j);
		team->getThread(j)->transfer.store(pop_top());
	}
	// update_status();
	request.store(NO_REQU);
}

// Keep executing tasks until m_stop = true && queue is empty
void spin(Thread_prio * a_thread)
{
    LOG(a_thread->m_id);
    // std::unique_lock<std::mutex> lck(a_thread->mtx);
	while (true) {
		/*
		if (a_thread->m_id != 0) {
			printf("hi %d\n", a_thread->m_id);
		}
		*/
		if (a_thread->ready_queue.empty()) {
			// printf("%d is empty\n", a_thread->m_id);
			// std::cout << "acquiring\n";
			a_thread->acquire();
			if (a_thread->team->endMe.load()) {
				// std::cout << "stopping\n";
				return;
			}
			// printf("%d isn't empty anymore hopefully\n", a_thread->m_id);
		} else {
			Task* tsk = a_thread->pop_bottom();
			// printf("popping. new size: %d\n", a_thread->ready_queue.size());
			a_thread->update_status();
			a_thread->communicate();
			// std::string str = std::to_string(a_thread->m_id);
			// str += " is about to run a task broseph\n";
			// str += "current # tasks: ";
			// str += std::to_string(a_thread->ready_queue.size());
			// str += "\n";
			// std::cout << str;
			// printf("get ready!!\n");
			bool shouldStop = (*tsk)(a_thread->m_id);
			// printf("not this time, friend\n");
			if (shouldStop) {
				// std::cout << "attempting to stop\n";
				a_thread->team->endMe.store(true);
			}
			// std::cout << "donezo\n";
			if (tsk->m_delete) delete tsk;
		}
	}
	/*
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
	*/
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
	int3 finish;

    struct Init {
        // How to initialize a task
        Init_task init;
        // Mapping from task index to thread id
        Map_task map;
		// The last node of the graph
		int3 idx;

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
    void async(int3, Task*, int);

    // Decrement the dependency counter and spawnspawn task if ready
    void decrement_wait_count(int3, int);

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

void Task_flow::async(int3 idx, Task * a_tsk, int m_id)
{
	bool shouldStop = idx == finish;
	/*
	std::string str = "running ";
	str += std::to_string(idx[0]);
	str += " ";
	str += std::to_string(idx[1]);
	str += " ";
	str += std::to_string(idx[2]);
	str += "\n";
	std::cout << str;
	*/
	// printf("trying to spawn %d %d %d\n", idx[0], idx[1], idx[2]);
    team->spawn(/*task map*/ /* m_task.map(idx) */ m_id, a_tsk, shouldStop);

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
    async(idx, t_, 0);
}

void Task_flow::decrement_wait_count(int3 idx, int m_id)
{
    Task * t_ = find_task(idx);

    // Decrement counter
    std::unique_lock<std::mutex> lck(t_->mtx);
    --(t_->m_wait_count);
    assert(t_->m_wait_count >= 0);
	
	// printf("decrementing %d %d %d to %d\n", idx[0], idx[1], idx[2], t_->m_wait_count);

    if (t_->m_wait_count == 0) { // task is ready to run
		
        lck.unlock();
        async(idx, t_, m_id);
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

#endif