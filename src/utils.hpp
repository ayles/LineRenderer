//
// Created by selya on 19.12.2019.
//

#ifndef LINERENDERER_UTILS_HPP
#define LINERENDERER_UTILS_HPP

#include <mutex>
#include <condition_variable>

namespace lr {

class Locker {
public:
    void Wait() {
        std::unique_lock lock(m);
        if (locked) {
            cv.wait(lock, [&]() { return !locked; });
        }
        locked = true;
    }

    void Wake() {
        std::unique_lock lock(m);
        locked = false;
        lock.unlock();
        cv.notify_all();
    }

    std::mutex &GetMutex() {
        return m;
    }

private:
    std::mutex m;
    std::condition_variable cv;
    bool locked = true;
};

}

#endif //LINERENDERER_UTILS_HPP
