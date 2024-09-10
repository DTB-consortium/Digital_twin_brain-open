#pragma once

#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>

namespace dtb {

template<typename T>
class BlockingQueue
{
public:
	BlockingQueue() = default;
	~BlockingQueue() = default;

	void push(const T& t)
	{
		{
			std::lock_guard<std::mutex> lock(mutex_);
			queue_.push(t);
		}
		condition_.notify_one();
	}
  
	T pop()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		condition_.wait(lock, [this]{return !queue_.empty();});
		T t(queue_.front());
		queue_.pop();
		return t;
	}

	bool try_peek(T* t)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (queue_.empty())
		{
			return false;
		}
		*t = queue_.front();
		return true;
	}
  
	bool try_pop(T* t)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		if (queue_.empty())
		{
			return false;
		}

		*t = queue_.front();
		queue_.pop();
		return true;
	}

  // Return element without removing it
	T peek()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		condition_.wait(lock, [this]{return !queue_.empty();});
		return queue_.front();
	}

	void clear()
	{
		std::queue<T> empty;
		std::unique_lock<std::mutex> lock(mutex_);
		std::swap(empty, queue_);
	}

	bool empty() const
	{
		std::unique_lock<std::mutex> lock(mutex_);
		return queue_.empty();
	}

	size_t size() const
	{
		std::unique_lock<std::mutex> lock(mutex_);
		return queue_.size();
	}

 protected:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;

  DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace dtb

