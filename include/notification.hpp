#pragma once

#include <stdint.h>
#include <cassert>
#include <atomic>      
#include <chrono>
#include <mutex>
#include <condition_variable>

namespace dtb {

class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {
    // In case the notification is being used to synchronize its own deletion,
    // force any prior notifier to leave its critical section before the object
    // is destroyed.
    std::unique_lock<std::mutex> lock(mutex_);
  }

  void Notify() {
    std::unique_lock<std::mutex> lock(mutex_);
    notified_.store(true, std::memory_order_release);
	lock.unlock();
    cv_.notify_one();
  }

  bool IsNotified() const {
    return notified_.load(std::memory_order_acquire);
  }

  void Wait() {
	std::unique_lock<std::mutex> lock(mutex_);
	while (!IsNotified()) {
		cv_.wait(lock);
	}

	notified_.store(false, std::memory_order_release);
  }

  bool WaitWithTimeout(int64_t timeout_in_us) 
  {
    bool notified = IsNotified();
    if (!notified) {
      std::unique_lock<std::mutex> lock(mutex_);
      do {
        notified = IsNotified();
      } while (!notified &&
               cv_.wait_for(lock, std::chrono::microseconds(timeout_in_us)) !=
                   std::cv_status::timeout);
    }
	
    return notified;
  }

 private:
  
  std::mutex mutex_;                    // protects mutations of notified_
  std::condition_variable cv_;       // signaled when notified_ becomes non-zero
  std::atomic<bool> notified_;  	 // mutations under mu_
};

}

